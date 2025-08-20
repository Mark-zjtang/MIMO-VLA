import json
import re
import os
os.environ['SDL_VIDEODRIVER'] = "dummy"
import sys
from typing import List
import pygame
import glob
from env_carla_obschannel import Env
import fire
import numpy as np
import torch
import transformers
import wandb
import pickle
import itertools
import random
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import has_length
from transformers.trainer_pt_utils import SequentialDistributedSampler
from utils.prompter import Prompter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import Dataset,  random_split
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers.trainer import is_datasets_available, RandomSampler
from transformers import CLIPProcessor, CLIPModel
from functools import partial
import datasets
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, PeftModel
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from torch import nn
from torch.nn import functional as F
from transformers import Trainer, GenerationConfig
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from finetune_mimovla_training import VlmDTModel, CustomDataset, CustomTrainer
from collections import OrderedDict
from torchvision.utils import make_grid, save_image
from action_tokenizer import ActionTokenizer


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

questions = [
    "What are you seeing/observing?",
    "What are you paying attention to and why?",
    "Are there any traffic lights? What’s the color of the traffic light?",
    "What’s your current speed and steering angle?",
    "What is your action and why?",
    "Summarize the current driving scenario at a high level.",
    "How are you going to drive in this situation and why?",
    "What’s the straight-line distance to the nearest car?",
    "What is the angle of the nearest car relative to your heading?",
    "Is there any lateral deviation from your driving route?",
    "What should be your next steering action?",
    "What should be your next acceleration command?",
    "Is there any moving object around you?",
    "Describe the position of the car relative to your heading.",
    "What is your current lateral position relative to your route?",
    "What would be a safe driving action given the detected car's details?",
    "What is the speed of the detected car?",
    "How far is the detected car from you?",
    "What angle should you adjust your steering to avoid collision?",
    "Why is it important to note the angle of the detected car?",
    "Is the detected car in motion?",
    "What should you be cautious of given the car’s position?",
    "What action should be taken to maintain alignment on your driving route?",
    "What considerations are necessary for the detected car's speed?",
    "What’s the importance of your current lateral position in planning the next action?",
    "What factors are influencing your next driving decision?",
    "Is there any obstacle directly ahead?",
    "How should you interpret the car’s angle for your steering decision?",
    "What immediate adjustments are necessary for safe driving?",
    "How does the detected car's speed impact your driving action?",
    "What should be your focus given the detected car’s proximity and angle?",
    "What safe action is suggested based on the current scenario?",
    "What should you avoid in this situation to prevent collision?",
    "Is there a need for a speed adjustment?",
    "How will your steering angle change based on the detected car’s angle?",
    "What should you consider for maintaining a safe path?",
    "How would you describe the current traffic conditions?",
    "What immediate action is necessary given your current lateral position?",
    "What factors need to be monitored to ensure safe navigation?",
    "Is the detected car influencing your path directly?",
    "What is the priority in adjusting your speed and direction?",
    ""
]

def tokenize_no_padding(prompt, cutoff_length, tokenizer, add_eos_token=True):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_length,
        padding=True,
        return_tensors=None,
    )
    eos_token_id = tokenizer.eos_token_id
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_length
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point,
                                 cutoff_length,
                                 prompter,
                                 tokenizer,
                                 input_name=None,
                                 train_on_inputs=True,
                                 padding=True,
                                 is_eval=False):
    if is_eval:
        full_prompt = prompter.generate_prompt(
            data_point["input"],
            data_point["Question"],
        )
    else:
        full_prompt = prompter.generate_prompt(
            data_point["input"],
            data_point["Question"],
            data_point["Answer"]
        )
    tokenized_full_prompt = tokenize_no_padding(full_prompt, cutoff_length, tokenizer)
    input_real_seq_length = len(tokenized_full_prompt["input_ids"])
    if padding:

        input_real_seq_length_padding = [0] * (cutoff_length - input_real_seq_length+2)
        tokenized_full_prompt["input_ids"].extend(input_real_seq_length_padding)

        tokenized_full_prompt["input_real_seq_length"] = input_real_seq_length
        if not is_eval:
            tokenized_full_prompt["inputs_obs"] = data_point["lidar_noground"]
            tokenized_full_prompt["action"] = data_point["action"]
            tokenized_full_prompt["sum_rewards"] = data_point["sum_rewards"]
        tokenized_full_prompt["attention_mask"].append(1)
        tokenized_full_prompt["attention_mask"].append(1)
        tokenized_full_prompt["attention_mask"].append(1)
        input_real_seq_length_padding_mask = [1] * (cutoff_length - input_real_seq_length - 3)
        tokenized_full_prompt["attention_mask"].extend(input_real_seq_length_padding_mask)
    else:
        tokenized_full_prompt["input_real_seq_length"] = input_real_seq_length
        if not is_eval:
            tokenized_full_prompt["inputs_obs"] = data_point["lidar_noground"]
            tokenized_full_prompt["action"] = data_point["action"]
            tokenized_full_prompt["sum_rewards"] = data_point["sum_rewards"]

    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["input"], data_point["Question"]
        )
        tokenized_user_prompt = tokenize_no_padding(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                     user_prompt_len:]
        # could be sped up, probably
    return tokenized_full_prompt

def main(
    load_8bit: bool = False,
    base_model: str = "alpaca-lora-carla/llama-7b-hf",
    data_path_pkl: str = "",
    lora_weights: str = "",
    prompt_template: str = "alpaca-lora-carla/templates/alpaca",
    cutoff_length: int = 424,
    output_dir2: str = "alpaca-lora-carla/original_lora_model",
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    id: str = "vlm_dt_eval_in_carla",
    seed: int = 1,  # Random seed
    env: str = 'carla-v1',  #Gym/Control Suite environment
    symbolic_env: bool = False,  # Symbolic features
    input_names: List[str] = ['lidar_noground'],  # Input_channel names
    mask_names: List[str] = [],  # Mask names
    max_episode_length: int = 2001,  # Max episode length
    action_repeat: int = 1,  # Action repeat
    episodes: int = 2000, # Total number of episodes
    bit_depth: int = 5,
    eval_in_carla: bool = False,
    render: bool = False,
    no_eval_answer: bool = False,
    carla_port: int = 2000,  # range in [2000, 9000]
    town: str = 'Town03',
    is_open_vla: bool = False,
    method: str = 'VLA4CD',
    text_nums : int = 100,
):
    print(
        f"Running with parameters:\n"
        f"load_8bit: {load_8bit}\n"
        f"base_model: {base_model}\n"
        f"lora_weights: {lora_weights}\n"
        f"prompt_template: {prompt_template}\n"
        f"cutoff_length: {cutoff_length}\n"
        f"output_dir2: {output_dir2}\n"
        f"server_name: {server_name}\n"
        f"share_gradio: {share_gradio}\n"
        f"id: {id}\n"
        f"seed: {seed}\n"
        f"env: {env}\n"
        f"symbolic_env: {symbolic_env}\n"
        f"input_names: {input_names}\n"
        f"mask_names: {mask_names}\n"
        f"max_episode_length: {max_episode_length}\n"
        f"action_repeat: {action_repeat}\n"
        f"episodes: {episodes}\n"
        f"bit_depth: {bit_depth}\n"
        f"eval_in_carla: {eval_in_carla}\n"
        f"render: {render}\n",
        f"no_eval_answer: {no_eval_answer}\n",
        f"carla_port: {carla_port}\n",
        f"town: {town}\n",
        f"is_open_vla: {is_open_vla}\n",
        f"data_path_pkl: {data_path_pkl}\n"

    )

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float32,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
    # model = prepare_model_for_int8_training(model)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    # Compute reward/collision_distance in carla
    global action_llm, predict_action
    # Load LoRA configuration if applicable
    lora_config_path = os.path.join(lora_weights, "adapter_config.json")
    if os.path.exists(lora_config_path):
        # from peft import LoraConfig, LoraModel
        lora_model_config = LoraConfig.from_json_file(lora_config_path)
        lora_model_config = LoraConfig(
            r=lora_model_config['r'],
            lora_alpha=lora_model_config['lora_alpha'],
            target_modules=lora_model_config['target_modules'],
            lora_dropout=lora_model_config['lora_dropout'],
            bias=lora_model_config['bias'],
            task_type=lora_model_config['task_type'],
            fan_in_fan_out=lora_model_config['fan_in_fan_out'],
            peft_type=lora_model_config['peft_type'],
        )
        custom_path = os.path.join(lora_weights, "custom_config.json")
        with open(custom_path, 'r') as json_file:
            custom_config = json.load(json_file)
        # Load LoRA-adapted weights
        lora_path = os.path.join(lora_weights, "pytorch_model.bin")
        if os.path.exists(lora_path):
            lora_state_dict1 = torch.load(lora_path, map_location="cuda")
            lora_state_dict2 = OrderedDict(lora_state_dict1)
            def replace_key_in_dict(d):
                new_dict = {}
                for key, value in d.items():
                    new_key = key.replace('lora_model.base_model',
                                          'base_model') if 'lora_model.base_model' in key else key
                    new_dict[new_key] = value
                return new_dict

            lora_state_dict3 = replace_key_in_dict(lora_state_dict2)
            lora_state_dict4 = OrderedDict(lora_state_dict3)
            model = get_peft_model(model, lora_model_config)
            model = VlmDTModel.from_pretrained(lora_weights, model, tokenizer)
            model.eval()
            input_types = custom_config['input_types']
            with open(data_path_pkl, 'rb') as f:
                data = pickle.load(f)
            eval_text_datasets = []

            def extract_answer(text):
                match = re.search(r'### answer:\s*(.*)', text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return None
            with open(f'{method}_eval_chatting_nums_{text_nums}.json', 'w') as f1:
                for i in tqdm(range(text_nums)):
                    new_input = data[i]["input"]
                    chosen_question = data[i]["Question"]
                    answer_label = data[i]["Answer"]
                    input_data_point = {"input": new_input, "Question": chosen_question}
                    infer_dataset = generate_and_tokenize_prompt(input_data_point, cutoff_length, prompter, tokenizer, input_name=input_names, padding=False, is_eval=True)
                    input_ids = torch.tensor(infer_dataset['input_ids'], device=device)
                    attention_mask = torch.tensor(infer_dataset['attention_mask'], device=device)
                    inputs_obs = torch.tensor(data[i]["lidar_noground"], device=device)
                    sum_rewards = torch.tensor(data[i]["reward"]).unsqueeze(0).to(device=device, dtype=torch.float32)
                    input_real_seq_length = torch.tensor(infer_dataset["input_real_seq_length"])
                    with torch.no_grad():
                        _, _, predict_action, predict_text_seq, input_ids_from_embeds,  _, \
                            = model(labels=None,
                                    input_ids=input_ids,
                                    ones=None,
                                    attention_mask=attention_mask,
                                    inputs_obs=inputs_obs,
                                    action=None,
                                    sum_rewards=sum_rewards,
                                    input_real_seq_length=input_real_seq_length,
                                    real_labels=None,
                                    # openvla_action=None,
                                    is_eval=True,  # setting False: not predict answer because much times
                                    no_eval_answer=no_eval_answer,
                                    image_embeds_clip=None,
                                    text_embeds_clip=None)
                        if not no_eval_answer:
                                predict_text = tokenizer.decode(predict_text_seq, skip_special_tokens=True)
                                print(f'input : {new_input}')
                                print(f'chose_question : {chosen_question}')
                                print(f'###=========predict_orignal_answer==========###:  {predict_text}')
                                print(f'###=========predict_answer==========###:  {extract_answer(predict_text)}')
                                print(f'###=========answer_label==========###:  {answer_label}')
                        else:
                             raise ValueError('Only predict text')
                        eval_text_datasets.append({"predict_answer": extract_answer(predict_text), "answer_label": answer_label})
                print('Successfully test')
                json.dump(eval_text_datasets, f1)



if __name__ == "__main__":
    fire.Fire(main)
