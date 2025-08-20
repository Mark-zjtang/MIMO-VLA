import json
import re
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import sys
from typing import List
import fire
import numpy as np
import torch
import transformers
import wandb
import pickle
import itertools
# import clip
from PIL import Image
from torch import nn, optim
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
from torch.utils.data import Dataset, random_split
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers.trainer import is_datasets_available, RandomSampler
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPTextConfig, CLIPTokenizer
from functools import partial
from pathlib import Path

torch.autograd.set_detect_anomaly(True)
# transformers.logging.set_verbosity_info()

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import datasets
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, PeftModel, LoraModel
)
from transformers import LlamaForCausalLM, LlamaTokenizer, CLIPProcessor, CLIPModel
from utils.prompter import Prompter
from torch import nn
from torch.nn import functional as F
from transformers import Trainer, GenerationConfig
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from torch.utils.data import Sampler, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random


class RandomSegmenter:
    def __init__(self, data, segment_length):
        if segment_length <= 0 or segment_length > len(data):
            raise ValueError("Segment length must be positive and less than or equal to the length of the data.")

        self.data = data
        self.segment_length = segment_length

    def generate_all_segments(self):
        return [self.data[i:i + self.segment_length] for i in range(len(self.data) - self.segment_length + 1)]

    def random_non_overlapping_split(self):
        all_segments = self.generate_all_segments()
        random.shuffle(all_segments)

        chosen_segments = []
        used_indices = set()

        for segment in all_segments:
            start_index = self.data.index(segment[0])
            if all(i not in used_indices for i in range(start_index, start_index + self.segment_length)):
                chosen_segments.append(segment)
                used_indices.update(range(start_index, start_index + self.segment_length))

        remainder_length = len(self.data) % self.segment_length
        if remainder_length > 0:
            remainder_segment = self.data[-remainder_length:]
            if chosen_segments:
                chosen_segments[-1].extend(remainder_segment)
            else:
                chosen_segments.append(remainder_segment)

        return chosen_segments

    def concatenate_segments(self, segments):

        return [item for segment in segments for item in segment]

    def get_randomized_array(self):

        segments = self.random_non_overlapping_split()
        return self.concatenate_segments(segments)

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            dataset: Optional[Dataset] = None,
            lengths: Optional[List[int]] = None,
            model_input_name: Optional[str] = None,
            generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator


    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        # indices = self.get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)

        sample_indices = True
        multi_sample = True
        segment_length = 1 # horizon =1, 2, 4, 8
        print(f'seg horizon: {segment_length}')
        if multi_sample:
            indices = torch.tensor(range(len(self.lengths)))
            indices = indices.numpy().tolist()
            segmenter = RandomSegmenter(indices, segment_length)
            result = segmenter.get_randomized_array()
        else:
            if sample_indices:
                shuffled_indices = torch.randperm(len(self.lengths))
                indices = shuffled_indices.numpy().tolist()
            else:
                indices = torch.tensor(range(len(self.lengths)))
                indices = indices.numpy().tolist()

        return iter(indices)


def check_is_nan(x):
    nan_mask = torch.isnan(x)
    nan_indices = torch.nonzero(nan_mask)
    assert not torch.any(torch.isnan(x))
    return nan_mask


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.attn = nn.Linear(feature_dim * 2, feature_dim)
        self.v = nn.Parameter(torch.rand(feature_dim))

    def forward(self, img_features, text_features):
        batch_size, img_len, feature_dim = img_features.size()
        text_len = text_features.size(1)

        # Repeat image features for each text token
        img_features_exp = img_features.unsqueeze(2).repeat(1, 1, text_len, 1)
        # Repeat text features for each image token
        text_features_exp = text_features.unsqueeze(1).repeat(1, img_len, 1, 1)

        # Concatenate along feature dimension
        concat_features = torch.cat((img_features_exp, text_features_exp), dim=-1)

        # Compute attention scores
        energy = torch.tanh(self.attn(concat_features))
        energy = torch.matmul(energy, self.v)
        attention = F.softmax(energy, dim=2)

        # Compute context vector
        context_vector = torch.einsum('bij,bijk->bik', attention, text_features_exp)

        return context_vector


class MultimodalModel(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MultimodalModel, self).__init__()
        self.attention = Attention(feature_dim)
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, img_features, text_features):
        context_vector = self.attention(img_features, text_features)

        # Pool over image tokens (mean pooling)
        pooled_vector = torch.mean(context_vector, dim=1)

        # Downstream task
        output = self.fc(pooled_vector)

        return output


class VlmDTModel(nn.Module):

    def __init__(self,
                 original_model,
                 input_types,
                 loss_type,
                 batch_size,
                 save_steps,
                 lora_config,
                 lora_target_modules,
                 writer,
                 lambda_action=1,
                 long_horizon=True,
                 img_patch_size=16,# default 16
                 lambda_smooth=1,
                 lambda_steer=1,
                 lambda_accer=1,
                 lambda_img=1,
                 horizon=8,
                 cutoff_length=424,
                 is_multi_frames=False,
                 is_multi_text_obs=False,
                 regular_action_loss=False,
                 is_change_text_feature=False,
                 shift_labels=False,
                 micro_batch_size=16,
                 global_step=0,
                 default_input_embedding_layer=True,
                 is_atten_fusion=False,
                 fc_fusion_layers=0,
                 is_embeds_clip=False,
                 tokenizer=None,
                 is_truncation_logits=False,
                 is_reconstruction=False,
                 is_multi_reconstruct=False):
        super(VlmDTModel, self).__init__()
        self.lora_model = original_model
        self.lora_config = lora_config
        self.long_horizon = long_horizon
        self.embedding_hidden_size = 4096
        self.hidden_size = 1024
        self.action_size = 2
        self.input_types = input_types
        self.loss_type = loss_type
        self.is_reconstruction = is_reconstruction
        self.img_patch_size = img_patch_size
        self.num_patches = (128 // self.img_patch_size) ** 2
        self.tokenizer_vocab_size = 32000
        self.split_obs_proj = nn.Conv2d(3, self.embedding_hidden_size, kernel_size=self.img_patch_size,
                                        stride=self.img_patch_size).to('cuda')
        self.inverse_split_obs_proj = nn.ConvTranspose2d(in_channels=self.embedding_hidden_size, out_channels=3, kernel_size=self.img_patch_size,
                                                         stride=self.img_patch_size).to('cuda')
        self.split_obs_position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.embedding_hidden_size)).to('cuda')
        self.text_embedding = nn.Embedding(self.tokenizer_vocab_size, self.embedding_hidden_size).to('cuda')
        self.custom_lm_head = nn.Linear(self.embedding_hidden_size, self.tokenizer_vocab_size, bias=False).to('cuda')
        self.actor_linear1 = nn.Linear(self.embedding_hidden_size, 2048).to('cuda')
        self.actor_linear2 = nn.Linear(2048, 1024).to('cuda')
        self.actor_linear3 = nn.Linear(1024, 512).to('cuda')
        self.actor_linear4 = nn.Linear(512, 256).to('cuda')
        self.actor_linear5 = nn.Linear(256, 128).to('cuda')
        self.actor_linear6 = nn.Linear(128, 64).to('cuda')
        self.actor_linear7 = nn.Linear(64, self.action_size).to('cuda')
        self.fc_reconstruction = nn.Linear(self.embedding_hidden_size, micro_batch_size*3*128*128).to('cuda')
        self.action_linear = nn.Linear(2, self.embedding_hidden_size).to('cuda')
        self.reward_linear = nn.Linear(1, self.embedding_hidden_size).to('cuda')
        self.full_fusion_linear1 = nn.Linear(self.embedding_hidden_size, self.embedding_hidden_size).to('cuda')
        self.full_fusion_linear2 = nn.Linear(self.embedding_hidden_size, self.embedding_hidden_size).to('cuda')
        self.lambda_smooth = lambda_smooth
        self.lambda_action = lambda_action
        self.lambda_accer = lambda_accer
        self.lambda_steer = lambda_steer
        self.lambda_img = lambda_img
        self.horizon = horizon
        self.cutoff_length = cutoff_length
        self.shift_labels = shift_labels
        self.is_multi_frames = is_multi_frames
        self.is_multi_text_obs = is_multi_text_obs
        self.is_multi_reconstruct = is_multi_reconstruct
        self.save_steps = save_steps
        self.batch_size = batch_size
        self.config = self.lora_model.config
        self.lora_target_modules = lora_target_modules
        self.prepare_inputs_for_generation = self.lora_model.prepare_inputs_for_generation
        self._keys_to_ignore_on_save = []  # Optional: specify keys to ignore during save
        self.writer = writer
        self.is_change_text_feature = is_change_text_feature
        self.micro_batch_size = micro_batch_size
        self.global_step = global_step
        self._keys_to_save = []  # Optional: specify keys to save
        self.default_input_embedding_layer = default_input_embedding_layer
        self.tokenizer = tokenizer
        ############### Add attention fusion to text and obs
        self.feature_dim = self.embedding_hidden_size
        self.attn = nn.Linear(self.feature_dim * 2, self.feature_dim).to('cuda')
        self.v = nn.Parameter(torch.rand(self.feature_dim)).to('cuda')
        self.is_atten_fusion = is_atten_fusion
        self.fc_fusion_layers = fc_fusion_layers
        self.attention_fc = nn.Linear(1, 1).to('cuda')
        self.is_embeds_clip = is_embeds_clip
        self.is_truncation_logits = is_truncation_logits
        self.regular_action_loss = regular_action_loss

    def forward(self,
                labels,
                input_ids,
                ones,
                attention_mask,
                inputs_obs,
                action,
                sum_rewards,
                input_real_seq_length,
                real_labels,
                image_embeds_clip,
                text_embeds_clip,
                is_eval=False,
                no_eval_answer=False):
        # get some labels
        global model_outputs, outputs_hidden_layers_ids
        multi_input_embeds, llm_length, obs_r_length\
            = self.get_multi_inputs_fusion( self.input_types,
                                            input_llm=input_ids,
                                            attention_mask=attention_mask,
                                            input_obs=inputs_obs,
                                            action=action,
                                            horizon=self.horizon,
                                            sum_rewards=sum_rewards,
                                            default_input_embedding_layer=self.default_input_embedding_layer,
                                            is_eval=is_eval,
                                            input_real_seq_length=input_real_seq_length,
                                            image_embeds_clip=image_embeds_clip,
                                            text_embeds_clip=text_embeds_clip)

        self.model_outputs = self.lora_model(**multi_input_embeds, output_hidden_states=True)
        actor_predict_actions, predict_text_logits, predict_text_output, \
        input_ids_from_embeds, img_reconstruction = self.generate_output(
                                model=self.lora_model,
                                input_ids=input_ids,
                                inputs_embeds=multi_input_embeds['inputs_embeds'],
                                attention_mask=multi_input_embeds['attention_mask'],
                                model_outputs=self.model_outputs,
                                input_types=self.input_types,
                                input_ids_real_seq_length=input_real_seq_length,
                                is_eval=is_eval,
                                is_reconstruct=self.is_reconstruction,
                                is_multi_reconstruct=self.is_multi_reconstruct,
                                no_eval_answer=no_eval_answer,
                                input_real_seq_length=input_real_seq_length)

        return llm_length, self.model_outputs, actor_predict_actions, predict_text_output, input_ids_from_embeds, obs_r_length, img_reconstruction


    def attention_index(self, text_features, img_features):
        batch_size, img_len, _ = img_features.shape
        text_len = text_features.shape[1]

        # Repeat image features for each text token
        img_features_exp = img_features.unsqueeze(2).repeat(1, 1, text_len, 1)

        # Repeat text features for each image token
        text_features_exp = text_features.unsqueeze(1).repeat(1, img_len, 1, 1)

        # Concatenate along feature dimension
        concat_features = torch.cat((text_features_exp, img_features_exp), dim=-1)

        # Compute attention scores
        energy = torch.tanh(self.attn(concat_features))
        energy = torch.einsum('bijl,l->bij', energy, self.v)

        attention_weights = F.softmax(energy, dim=2)

        # Compute context vector
        context_vector = torch.einsum('bijk,bijk->bij', attention_weights.unsqueeze(-1), text_features_exp)
        return context_vector

    def multimodal_inference(self, text_features, img_features):
        """
        Args:
            img_features (torch.Tensor): Tensor of shape [batch_size, img_seq_len, feature_dim]
            text_features (torch.Tensor): Tensor of shape [batch_size, text_seq_len, feature_dim]
            feature_dim (int): Dimensionality of the feature vectors
            output_dim (int): Dimensionality of the output

        Returns:
            output (torch.Tensor): Output tensor from the multimodal model
        """
        # Apply attention to align features
        context_vector = self.attention_index(img_features, text_features)
        self.attention_fc = nn.Linear(context_vector.shape[-1], self.feature_dim).to('cuda')
        image_in_contex = self.attention_fc(context_vector)
        combined_features = torch.cat((img_features, image_in_contex), dim=1)

        return combined_features

    def get_multi_inputs_fusion(self,
                                input_types,
                                input_llm=None,
                                attention_mask=None,
                                input_obs=None,
                                horizon=None,
                                action=None,
                                sum_rewards=None,
                                default_input_embedding_layer=True,
                                is_eval=False,
                                input_real_seq_length=None,
                                image_embeds_clip=None,
                                text_embeds_clip=None,
                                is_multi_frames=None):

        global input_obs_r, input_obs_r_mask, obs_r_length, llm_input_embedding, input_sum_reward_features, input_obs_r_llm_action_final_ids, input_obs_r_embedding, input_obs_r_action_final, input_llm_embedding, llm_length
        if input_types == ['obs'] or input_types == ['text'] or input_types == ['obs', 'text']:
            # The fusion of inputs_llm and inputs_obs_ra
            if input_obs is not None:
                input_obs_features = self.split_obs_image(input_obs.unsqueeze(dim=0))
                input_sum_reward_features = self.reward_linear(sum_rewards)
                input_sum_reward_features = input_sum_reward_features.unsqueeze(dim=1)
                if input_obs_features.dim() != 2:
                    input_obs_r = torch.cat((input_obs_features, input_sum_reward_features), dim=1)
                else:
                    if is_eval:
                        input_sum_reward_features = input_sum_reward_features.permute(1, 0)
                        input_obs_r = torch.cat(
                            (input_obs_features.unsqueeze(dim=0), input_sum_reward_features.unsqueeze(dim=0)), dim=1)
                        attention_mask = attention_mask.unsqueeze(dim=0)
                        input_llm = input_llm.unsqueeze(dim=0)
                        attention_mask = attention_mask[:, : input_real_seq_length]
                    else:
                        input_obs_r = torch.cat((input_obs_features.unsqueeze(dim=0), input_sum_reward_features), dim=1)
                # getting input_obs_reward_action sequence : obs, total_rewards sequence from original inputs
                obs_r_length = input_obs_r.shape[1]
                input_obs_r_mask = torch.ones(input_obs_r.shape[0], input_obs_r.shape[1])

            if input_llm is not None:
                if input_llm.dim() == 1:
                    input_llm = input_llm.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if is_eval:
                    input_llm = input_llm[:, :input_real_seq_length]
                if default_input_embedding_layer:
                    get_input_embedding = self.lora_model.get_input_embeddings()
                    llm_input_embedding = get_input_embedding(input_llm)
                else:
                    llm_input_embedding = self.text_embedding(input_llm)

            if len(input_types) == 2:

                if self.is_atten_fusion:
                    if input_obs.size(0) != 1:
                        input_obs_r_llm_action_final = torch.stack(tuple([self.multimodal_inference(img_features=input_obs_r_i, text_features=llm_input_embedding_i)])
                                                                    for input_obs_r_i, llm_input_embedding_i in zip(input_obs_r, llm_input_embedding))
                    else:
                        input_obs_r_llm_action_final = self.multimodal_inference(img_features=input_obs_r, text_features=llm_input_embedding)

                elif self.fc_fusion_layers == 1:
                    input_obs_r_llm_action_final = torch.cat((input_obs_r.to('cuda'), llm_input_embedding), dim=1)
                    input_obs_r_llm_action_final = self.full_fusion_linear1(input_obs_r_llm_action_final)

                elif self.fc_fusion_layers == 2:
                    input_obs_r_llm_action_final = torch.cat((input_obs_r.to('cuda'), llm_input_embedding), dim=1)
                    input_obs_r_llm_action_final = self.full_fusion_linear1(input_obs_r_llm_action_final)
                    input_obs_r_llm_action_final = self.full_fusion_linear2(input_obs_r_llm_action_final)

                elif self.is_embeds_clip:
                    input_obs_r_llm_action_final = torch.cat((image_embeds_clip, input_sum_reward_features, text_embeds_clip), dim=1)

                elif self.is_multi_frames:
                    input_action_features = self.action_linear(action.to('cuda'))
                    input_action_features = input_action_features.unsqueeze(dim=1)
                    input_obs_r_llm_action_final = torch.cat((input_obs_r.to('cuda'), llm_input_embedding), dim=1)
                    input_obs_r_llm_action_final = torch.cat((input_obs_r_llm_action_final, input_action_features), dim=1)
                    if input_obs_r_llm_action_final.shape[0] % horizon == 0:
                        input_obs_r_llm_action_final = input_obs_r_llm_action_final.reshape(input_obs_r_llm_action_final.shape[0] // horizon, -1, self.embedding_hidden_size)
                    else:
                        input_obs_r_llm_action_final = input_obs_r_llm_action_final[-1].unsqueeze(0)
                    input_obs_r_llm_action_final = input_obs_r_llm_action_final[:, :-1]
                else:
                    input_obs_r_llm_action_final = torch.cat((input_obs_r.to('cuda'), llm_input_embedding), dim=1)

                if self.is_embeds_clip:
                    input_obs_r_llm_real_seq_action_padding_final_mask = torch.cat((torch.ones(image_embeds_clip.shape[0], image_embeds_clip.shape[1]+1).to('cuda'),
                                                                                    attention_mask), dim=1)
                elif self.is_multi_frames:
                    input_obs_r_llm_real_seq_action_padding_final_mask = torch.ones(input_obs_r_llm_action_final.shape[0], input_obs_r_llm_action_final.shape[1])

                else:
                    input_obs_r_llm_real_seq_action_padding_final_mask = torch.cat((input_obs_r_mask.to('cuda'), attention_mask), dim=1)

                input_obs_r_llm_embedding = {"inputs_embeds": input_obs_r_llm_action_final,
                                             "attention_mask": input_obs_r_llm_real_seq_action_padding_final_mask}

                return input_obs_r_llm_embedding, None, obs_r_length

            elif input_types == ['obs']:
                if not self.is_multi_frames:
                    input_a = torch.zeros(input_obs_r.shape[0], 2)
                    input_action_features = self.action_linear(input_a.to('cuda'))
                    input_action_features = input_action_features.unsqueeze(dim=1)
                    input_obs_r_a = torch.cat((input_obs_r, input_action_features.to('cuda')), dim=1)
                    input_obs_r_a_mask = torch.ones(input_obs_r_a.shape[0],
                                                    input_obs_r_a.shape[1])
                    input_obs_r_embedding = {"inputs_embeds": input_obs_r_a,
                                             "attention_mask": input_obs_r_a_mask}
                else:
                    input_a = torch.zeros(input_obs_r.shape[0], 2)
                    input_action_features = self.action_linear(input_a.to('cuda'))
                    input_action_features = input_action_features.unsqueeze(dim=1)
                    input_obs_r_action_final = torch.cat((input_obs_r.to('cuda'), input_action_features), dim=1)
                    if input_obs_r_action_final.shape[0] % horizon == 0:
                        input_obs_r_action_final = input_obs_r_action_final.reshape(
                            input_obs_r_action_final.shape[0] // horizon, -1, self.embedding_hidden_size)
                    else:
                        input_obs_r_action_final = input_obs_r_action_final[-1].unsqueeze(0)
                    input_obs_r_action_final = input_obs_r_action_final[:, :-1]
                    input_obs_r_a_mask = torch.ones(input_obs_r_action_final.shape[0],
                                                    input_obs_r_action_final.shape[1])
                    input_obs_r_embedding = {"inputs_embeds": input_obs_r_action_final,
                                             "attention_mask": input_obs_r_a_mask}


                return input_obs_r_embedding, None, obs_r_length

            elif input_types == ['text']:
                if not self.is_multi_frames:
                    llm_length = llm_input_embedding.shape[1] - 1
                    input_llm_embedding = {"inputs_embeds": llm_input_embedding,
                                           "attention_mask": attention_mask}
                else:
                    llm_length = llm_input_embedding.shape[1] - 1
                    input_a = torch.zeros(llm_input_embedding.shape[0], 2)
                    input_action_features = self.action_linear(input_a.to('cuda'))
                    input_action_features = input_action_features.unsqueeze(dim=1)
                    input_r_text = torch.cat((llm_input_embedding, input_sum_reward_features), dim=1)
                    input_r_text_action_final = torch.cat((input_r_text.to('cuda'), input_action_features), dim=1)
                    if input_r_text_action_final.shape[0] % horizon == 0:
                        input_r_text_action_final = input_r_text_action_final.reshape(
                            input_r_text_action_final.shape[0] // horizon, -1, self.embedding_hidden_size)
                    else:
                        input_r_text_action_final = input_r_text_action_final[-1].unsqueeze(0)
                    input_r_text_action_final = input_r_text_action_final[:, :-1]
                    input_r_text_a_mask = torch.ones(input_r_text_action_final.shape[0],
                                                    input_r_text_action_final.shape[1])
                    input_llm_embedding = {"inputs_embeds": input_r_text_action_final,
                                             "attention_mask": input_r_text_a_mask}

                return input_llm_embedding, llm_length, None

            else:

                raise TypeError(f"Input types is not in input_list")
        else:
            raise TypeError(f"Input types is not in input_list")

    def tie_weights(self):
        # Define this method if it's necessary for your model
        pass

    def split_obs_image(self, obs_image):
        obs_image = obs_image.squeeze(dim=0)
        obs_image = obs_image.squeeze(dim=1)
        x = self.split_obs_proj(obs_image)  # (B, embed_dim, num_patches_w, num_patches_h)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.split_obs_position_embedding.to('cuda')  # add position embedding
        x = x.squeeze(dim=0)
        return x

    def inverse_split_obs_image(self, final_obs_image):
        final_obs_image = final_obs_image.to('cuda') + (-self.split_obs_position_embedding.to('cuda'))  # Remove position embedding
        # (B, embed_dim, num_patches_w, num_patches_h)
        final_obs_image = final_obs_image.view(-1, self.embedding_hidden_size, self.num_patches//8, self.num_patches//8)
        # obs_image
        obs_image = self.inverse_split_obs_proj(final_obs_image)

        return obs_image

    def generate_output(self,
                        model=None,
                        input_ids=None,
                        inputs_embeds=None,
                        attention_mask=None,
                        model_outputs=None,
                        input_types=None,
                        input_ids_real_seq_length=None,
                        is_eval=False,
                        is_reconstruct=False,
                        is_multi_reconstruct=False,
                        no_eval_answer=False,
                        input_real_seq_length=None):
        global predict_actions, input_ids_from_embeds
        outputs_hidden_layers = model_outputs.hidden_states[-1]
        if is_eval:
            outputs_hidden_layers = outputs_hidden_layers.to(dtype=torch.float32)
        if is_reconstruct:
            img_reconstruct = self.inverse_split_obs_image(outputs_hidden_layers[:, : self.num_patches])
        elif is_multi_reconstruct and outputs_hidden_layers.shape[0] == 1:
            img_reconstruct = self.inverse_split_obs_image(outputs_hidden_layers[:, : self.num_patches])
        elif is_multi_reconstruct and self.horizon == 2 and outputs_hidden_layers.shape[0] > 1:
            stack_reimage_outputs_hiddens = torch.stack([outputs_hidden_layers[:, : self.num_patches],
                                             outputs_hidden_layers[:, self.num_patches + 2: self.num_patches*2 + 2],
                                             ])
            stack_reimage_outputs_hiddens = stack_reimage_outputs_hiddens.view(-1, self.num_patches, self.embedding_hidden_size)
            img_reconstruct = self.inverse_split_obs_image(stack_reimage_outputs_hiddens)
        elif is_multi_reconstruct and self.horizon == 4 and outputs_hidden_layers.shape[0] > 1:
            stack_reimage_outputs_hiddens = torch.stack([outputs_hidden_layers[:, : self.num_patches],
                                             outputs_hidden_layers[:, self.num_patches + 2: self.num_patches*2 + 2],
                                             outputs_hidden_layers[:, self.num_patches*2 + 2: self.num_patches*3 + 2],
                                             outputs_hidden_layers[:, self.num_patches*3 + 2: self.num_patches * 4 + 2]
                                             ])
            stack_reimage_outputs_hiddens = stack_reimage_outputs_hiddens.view(-1, self.num_patches, self.embedding_hidden_size)
            img_reconstruct = self.inverse_split_obs_image(stack_reimage_outputs_hiddens)
        elif is_multi_reconstruct and self.horizon == 8:
            stack_reimage_outputs_hiddens = torch.stack([outputs_hidden_layers[:, : self.num_patches],
                                             outputs_hidden_layers[:, self.num_patches + 2: self.num_patches*2 + 2],
                                             outputs_hidden_layers[:, self.num_patches*2 + 2: self.num_patches*3 + 2],
                                             outputs_hidden_layers[:, self.num_patches*3 + 2: self.num_patches * 4 + 2],
                                             outputs_hidden_layers[:, self.num_patches * 4 + 2: self.num_patches * 5 + 2],
                                             outputs_hidden_layers[:, self.num_patches * 5 + 2: self.num_patches * 6 + 2],
                                             outputs_hidden_layers[:, self.num_patches * 6 + 2: self.num_patches * 7 + 2],
                                             outputs_hidden_layers[:, self.num_patches * 7 + 2: self.num_patches * 8 + 2],
                                             ])
            stack_reimage_outputs_hiddens = stack_reimage_outputs_hiddens.view(-1, self.num_patches,
                                                                               self.embedding_hidden_size)
            img_reconstruct = self.inverse_split_obs_image(stack_reimage_outputs_hiddens)
        else:
            img_reconstruct = None
            print(f'No img_reconstruct')
        multi_batch_size = self.micro_batch_size // self.horizon
        if outputs_hidden_layers.shape[0] == multi_batch_size:
            if multi_batch_size == 1:
                input_ids_real_seq_length = input_ids_real_seq_length[-1]
            else:
                input_ids_real_seq_length = input_ids_real_seq_length.view(input_ids_real_seq_length.shape[0] // self.horizon, self.horizon)[:, -1]
        else:
            input_ids_real_seq_length = input_ids_real_seq_length

        if input_types == ['obs']:
            outputs_hidden_layers_action = outputs_hidden_layers[:, -1]
            outputs_hidden_layers_text_features = None

        elif input_types == ['text']:
            if outputs_hidden_layers.size(0) != 1:
                outputs_hidden_layers_action = torch.stack(
                    [outputs_hidden_layers[i, input_ids_real_seq_length[i]] for i in
                     range(outputs_hidden_layers.size(0))]
                )
            else:
                if is_eval:
                    outputs_hidden_layers_action = outputs_hidden_layers[0, input_ids_real_seq_length - 1]
                else:
                    if input_ids_real_seq_length.dim() >= 2:
                        if input_ids_real_seq_length.shape[0] == 1:
                            outputs_hidden_layers_action = torch.stack(
                                [outputs_hidden_layers[0, self.num_patches + 1 + int(input_ids_real_seq_length.item())]]
                            )
                        else:
                            outputs_hidden_layers_action = torch.stack(
                                [outputs_hidden_layers[0, self.num_patches + 1 + input_ids_real_seq_length.shape[0]]]
                            )
                    else:
                        outputs_hidden_layers_action = torch.stack(
                            [outputs_hidden_layers[0, self.num_patches + 1 + 1]]
                        )

            outputs_hidden_layers_text_features = outputs_hidden_layers

        elif len(input_types) == 2:
            if outputs_hidden_layers.size(0) != 1:
                outputs_hidden_layers_action = torch.stack(
                    [outputs_hidden_layers[i, self.num_patches + 1 + input_ids_real_seq_length[i]] for i in
                     range(outputs_hidden_layers.size(0))]
                )
            else:
                if is_eval:
                    outputs_hidden_layers_action = outputs_hidden_layers[
                        0, self.num_patches + 1 + input_ids_real_seq_length - 1]
                else:
                    if input_ids_real_seq_length.dim() >= 2:
                        if input_ids_real_seq_length.shape[0] == 1:
                            outputs_hidden_layers_action = torch.stack(
                                [outputs_hidden_layers[0, self.num_patches + 1 + int(input_ids_real_seq_length.item())]]
                            )
                        else:
                            outputs_hidden_layers_action = torch.stack(
                                [outputs_hidden_layers[0, self.num_patches + 1 + input_ids_real_seq_length.shape[0]]]
                            )
                    else:
                        outputs_hidden_layers_action = torch.stack(
                            [outputs_hidden_layers[0, self.num_patches + 1 + 1]]
                        )

            outputs_hidden_layers_text_features = outputs_hidden_layers
        else:
            raise TypeError(f"Input types is not in input_list")

        predict_actions = self.actor_linear1(outputs_hidden_layers_action.to('cuda'))
        predict_actions = self.actor_linear2(predict_actions).to('cuda').to(dtype=torch.float32)
        predict_actions = self.actor_linear3(predict_actions).to('cuda').to(dtype=torch.float32)
        predict_actions = self.actor_linear4(predict_actions).to('cuda').to(dtype=torch.float32)
        predict_actions = self.actor_linear5(predict_actions).to('cuda').to(dtype=torch.float32)
        predict_actions = self.actor_linear6(predict_actions).to('cuda').to(dtype=torch.float32)
        predict_actions = self.actor_linear7(predict_actions).to('cuda').to(dtype=torch.float32)

        if self.is_change_text_feature and outputs_hidden_layers_text_features is not None:
            predict_text_logits = self.custom_lm_head(outputs_hidden_layers_text_features)
        else:
            predict_text_logits = None

        # if input_types is ["obs", "text"] or ["text"]:
        if input_types is ["obs", "text"] or ["text"] and is_eval and not no_eval_answer:
            logits = model_outputs['logits']
            embedding_matrix = self.lora_model.get_input_embeddings().weight  # shape: (vocab_size, embedding_dim)
            input_ids_from_embeds = []
            for embed in inputs_embeds[0]:
                cos_sim = torch.nn.functional.cosine_similarity(embed.unsqueeze(0), embedding_matrix)
                closest_word_id = torch.argmax(cos_sim).item()
                input_ids_from_embeds.append(closest_word_id)
            input_ids_from_embeds = torch.tensor(input_ids_from_embeds).unsqueeze(dim=0)  # shape: (1, seq_len)
            decoder_input_from_embeds = self.tokenizer.decode(input_ids.tolist(),
                                                              skip_special_tokens=True)
            predict_output = self.generate(
                                input_types=input_types,
                                input_ids=input_ids.to('cuda'),
                                attention_mask=attention_mask,
                                input_embeds=inputs_embeds,
                                input_real_seq_length=input_real_seq_length)
            predict_output = predict_output.sequences[0]
        else:
            predict_output = None
            input_ids_from_embeds = None


        return predict_actions, predict_text_logits, predict_output, input_ids_from_embeds, img_reconstruct

    def generate(self,
                 input_types=None,
                 input_embeds=None,
                 input_ids=None,
                 attention_mask=None,
                 position_ids=None,
                 tokenizer=None,
                 input_real_seq_length=None,
                 temperature=0.05,
                 top_p=0.75,
                 top_k=40,
                 num_beams=4,
                 max_new_tokens=30,
                 **kwargs):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            bos_token_id=0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # do_sample=False,
            **kwargs,
        )
        with torch.no_grad():

            if len(input_types) == 2:
                input_embeds = input_embeds[:, 65:-2]
            else:
                input_embeds = input_embeds[:, :-2]
            generation_output = self.lora_model.generate(
                input_ids=input_ids.unsqueeze(0)[:, :-2],
                # input_ids=None,
                # inputs_embeds=input_embeds,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                num_return_sequences=1,
                position_ids=position_ids,
            )

            return generation_output

    def generate_from_embeds(self,
                             input_types,
                             model,
                             input_ids,
                             input_embeds,
                             attention_mask,
                             generation_config=None,
                             max_length=20,
                             custom_generation_config=False):

        temperature = 0.05
        top_p = 0.9
        top_k = 40
        num_beams = 4
        if custom_generation_config:
            # if input_types == 2:
            #     input_embeds = input_embeds[:, 6:]
            #     attention_mask = input_embeds[: 65:]
            generated_ids = input_ids[:-2].unsqueeze(0)
            model.eval()
            for _ in range(max_length):
                generated_embeds = model.get_input_embeddings()(generated_ids)
                embeds = torch.cat((input_embeds, generated_embeds), dim=1)
                with torch.no_grad():
                    # outputs = model(inputs_embeds=embeds, output_hidden_states=True)
                    outputs = model(input_ids=generated_ids, output_hidden_states=True)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        else:
            if generation_config is None:
                generation_config = GenerationConfig()

            temperature = generation_config.temperature
            top_p = generation_config.top_p
            top_k = generation_config.top_k
            num_beams = generation_config.num_beams
            batch_size = input_embeds.size(0)
            generated_ids = input_ids[-2].unsqueeze(0).unsqueeze(1)

            beam_scores = None
            if num_beams > 1:
                beam_scores = torch.zeros((batch_size * num_beams), dtype=torch.float).to(input_embeds.device)
                beam_scores[:batch_size] = 0
                beam_scores[batch_size:] = -1e9

            for index in range(max_length):
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < min_top_k,
                                                    torch.tensor(float('-inf')).to(next_token_logits.device),
                                                    next_token_logits)

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # Ensure indices are of type int64
                    sorted_indices_to_remove = sorted_indices_to_remove.long()
                    indices_to_remove = sorted_indices.scatter(1, sorted_indices_to_remove, 1)

                    # Create a boolean mask
                    mask = indices_to_remove.bool()
                    next_token_logits = next_token_logits.masked_fill(mask, float('-inf'))

                if num_beams > 1:
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
                    next_token_scores = next_token_scores.view(batch_size, num_beams * model.config.vocab_size)

                    next_token_scores, next_tokens = torch.topk(next_token_scores, num_beams, dim=1, largest=True,
                                                                sorted=True)
                    next_tokens = next_tokens % model.config.vocab_size

                    beam_scores = next_token_scores.view(-1)

                    input_embeds = input_embeds.repeat_interleave(num_beams, dim=0)
                    generated_ids = generated_ids.repeat_interleave(num_beams, dim=0)
                else:
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_tokens = torch.argmax(next_token_scores, dim=-1, keepdim=True)

                # Ensure correct dimensions for next_tokens
                next_tokens = next_tokens.view(-1,
                                               1)  # Flatten batch dimension and ensure shape [batch_size * num_beams, 1
                # Concatenate `generated_ids` and `next_tokens`
                generated_ids = torch.cat((generated_ids, next_tokens), dim=1)
                next_token_embeds = model.get_input_embeddings()(next_tokens)
                input_embeds = torch.cat((input_embeds, next_token_embeds), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones(1, 1).to('cuda')), dim=1)
        return generated_ids

    def save_custom_state_dict_parameters(self):
        custom_layers_parameters = {
            "long_horizon": self.long_horizon,
            "input_types": self.input_types,
            "loss_type": self.loss_type,
            "batch_size": self.batch_size,
            "save_steps": self.save_steps,
            "lambda_smooth": self.lambda_smooth,
            "lambda_action": self.lambda_action,
            "img_patch_size": self.img_patch_size,
            "is_change_text_feature": self.is_change_text_feature,
            "default_input_embedding_layer": self.default_input_embedding_layer,
            "is_attention_fusion": self.is_atten_fusion,
            "fc_fusion_layers": self.fc_fusion_layers,
            "is_multi_frames": self.is_multi_frames,
            "shift_labels": self.shift_labels
        }
        custom_layers_state_dict = {
            "split_obs_proj": self.split_obs_proj.state_dict(),
            "split_obs_position_embedding": self.split_obs_position_embedding,
            "text_embedding": self.text_embedding.state_dict(),
            "custom_lm_head": self.custom_lm_head.state_dict(),
            "actor_linear1": self.actor_linear1.state_dict(),
            "actor_linear2": self.actor_linear2.state_dict(),
            "actor_linear3": self.actor_linear3.state_dict(),
            "actor_linear4": self.actor_linear4.state_dict(),
            "actor_linear5": self.actor_linear5.state_dict(),
            "actor_linear6": self.actor_linear6.state_dict(),
            "actor_linear7": self.actor_linear7.state_dict(),
            "action_linear": self.action_linear.state_dict(),
            "reward_linear": self.reward_linear.state_dict(),
            "full_fusion_linear1": self.full_fusion_linear1.state_dict(),
            "full_fusion_linear2": self.full_fusion_linear2.state_dict(),
            "attn": self.attn.state_dict(),
            "v": self.v,
            "attention_fc": self.attention_fc.state_dict()

        }

        return custom_layers_parameters, custom_layers_state_dict

    def load_from_custom_layers_state_dict(self, output_dir):
        custom_layers_path = os.path.join(output_dir, "custom_layers.bin")
        if os.path.exists(custom_layers_path):
            custom_layers_state_dict = torch.load(custom_layers_path, map_location="cuda")
            self.split_obs_position_embedding = custom_layers_state_dict['split_obs_position_embedding']
            self.split_obs_proj.load_state_dict(custom_layers_state_dict['split_obs_proj'])
            self.text_embedding.load_state_dict(custom_layers_state_dict['text_embedding'])
            self.custom_lm_head.load_state_dict(custom_layers_state_dict['custom_lm_head'])
            self.actor_linear1.load_state_dict(custom_layers_state_dict['actor_linear1'])
            self.actor_linear2.load_state_dict(custom_layers_state_dict['actor_linear2'])
            self.actor_linear3.load_state_dict(custom_layers_state_dict['actor_linear3'])
            self.actor_linear4.load_state_dict(custom_layers_state_dict['actor_linear4'])
            self.actor_linear5.load_state_dict(custom_layers_state_dict['actor_linear5'])
            self.actor_linear6.load_state_dict(custom_layers_state_dict['actor_linear6'])
            self.actor_linear7.load_state_dict(custom_layers_state_dict['actor_linear7'])
            self.reward_linear.load_state_dict(custom_layers_state_dict['reward_linear'])
        else:
            raise ValueError(f'Not find custom_layers.bin in {custom_layers_path}')

    def extract_lora_keywords(self, state_dict, num_layers=32):
        lora_prefixes = ["lora_A", "lora_B"]
        lora_module1, lora_module2 = None, None
        lora_target_modules1 = {'q_proj', 'v_proj', 'k_proj', 'o_proj'}
        lora_target_modules2 = {'gate_proj', 'up_proj', 'down_proj'}
        if set(self.lora_target_modules).issubset(lora_target_modules1) and len(self.lora_target_modules) >= 1:
            lora_module1 = ['self_attn.{}'.format(target_module) for target_module in self.lora_target_modules]
        elif set(self.lora_target_modules).issubset(lora_target_modules2) and len(self.lora_target_modules) >= 1:
            lora_module2 = ['mlp.{}'.format(target_module) for target_module in self.lora_target_modules]
        elif not set(self.lora_target_modules).issubset(lora_target_modules2) and len(self.lora_target_modules) >= 1:
            lora_module2 = None
        elif len(self.lora_target_modules) == 0:
            print('There is no lora target_modules')
        else:
            print(f"Lora_target is not in {lora_target_modules1 + lora_target_modules2}")
            raise ValueError
        if lora_module2 is not None:
            lora_module = lora_module1 + lora_module2
        else:
            lora_module = lora_module1
        lora_model_keywords = []
        for layer_num in range(num_layers):
            for prefix in lora_prefixes:
                for module in lora_module:
                    param_name = "base_model.model.model.layers.{}.{}.{}.weight".format(layer_num, module, prefix)
                    if param_name in state_dict:
                        lora_model_keywords.append(param_name)
        for k, v in state_dict.items():
            if k in lora_model_keywords:
                self._keys_to_save.append(k)

        return self._keys_to_save

    def save_pretrained(self, output_dir: Optional[str] = None):
        # Save Custom_layers parameter
        custom_layers_parameters, custom_layers_state_dict = self.save_custom_state_dict_parameters()
        with open(os.path.join(output_dir, "custom_config.json"), 'w') as json_file:
            json.dump(custom_layers_parameters, json_file, indent=4)
        # Save LoRA-adapted weights
        if self.lora_config:
            model_keys_to_save = self.extract_lora_keywords(self.lora_model.state_dict())
            torch.save({k: v for k, v in self.lora_model.state_dict().items() if k in model_keys_to_save},
                       os.path.join(output_dir, "pytorch_model.bin"))
            torch.save(custom_layers_state_dict, os.path.join(output_dir, "custom_layers.bin"))
            self.lora_config.save_pretrained(output_dir)
            self._keys_to_save = []
        else:
            raise ValueError("LoRA configuration not found in the model. Ensure LoRA is properly configured.")

    def modify_state_dict_keys(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('lora_model.base_model.model.model.layers.'):
                new_key = key.replace('lora_model.base_model.', 'lora_model.')
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    @classmethod
    def from_pretrained(cls, output_dir, lora_model, tokenizer):
        # Load LoRA configuration if applicable
        lora_config_path = os.path.join(output_dir, "adapter_config.json")
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
            custom_path = os.path.join(output_dir, "custom_config.json")
            with open(custom_path, 'r') as json_file:
                custom_config = json.load(json_file)
            # Load LoRA-adapted weights
            lora_path = os.path.join(output_dir, "pytorch_model.bin")
            if os.path.exists(lora_path):
                lora_state_dict = torch.load(lora_path, map_location="cuda")
                output_log = os.path.join(f"{output_dir}/training_logs")
                os.makedirs(output_log, exist_ok=True)
                writer = SummaryWriter(output_log)
                model = cls(original_model=lora_model,
                            input_types=custom_config['input_types'],
                            # input_types=['text'],
                            loss_type=custom_config['loss_type'],
                            batch_size=custom_config['batch_size'],
                            save_steps=custom_config['save_steps'],
                            lora_config=lora_model_config,
                            lora_target_modules=lora_model_config.target_modules,
                            writer=writer,
                            img_patch_size=custom_config['img_patch_size'],
                            is_change_text_feature=custom_config['is_change_text_feature'],
                            default_input_embedding_layer=custom_config['default_input_embedding_layer'],
                            is_atten_fusion=False,
                            fc_fusion_layers=False,
                            is_embeds_clip=False,
                            lambda_action=custom_config['lambda_action'],
                            lambda_smooth=custom_config['lambda_smooth'],
                            lambda_steer=1,
                            lambda_accer=1,
                            regular_action_loss=True,
                            is_truncation_logits=True,
                            # is_multi_frames=custom_config['is_multi_frames'], # Training
                            is_multi_frames=False,  # Evaluating
                            # shift_labels=custom_config['shift_labels']

                            )
                model_state_dict = {}
                for key, value in lora_state_dict.items():
                    parts = key.split('.')
                    new_key = 'lora_model.' + '.'.join(parts[:-1]) + '.default.' + parts[-1]
                    model_state_dict[new_key] = value
                model_state_dict = OrderedDict(model_state_dict)
                model.load_state_dict(model_state_dict, strict=False)
                model.load_from_custom_layers_state_dict(output_dir=output_dir)
            else:
                raise ValueError("Not find pytorch_model.bin")
            model.tokenizer = tokenizer
            return model
        else:
            raise ValueError("Not find adapter_config.json")


class CustomTrainer(Trainer):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
            # return SequentialDistributedSampler(
            #     self.args.train_batch_size * self.args.gradient_accumulation_steps,
            # )
        else:
            return RandomSampler(self.train_dataset)

    def label_smoother_mse_loss(self,
                                model_outputs=None,
                                labels=None,
                                action_labels=None,
                                actor_predict_actions=None,
                                predict_text_logits=None,
                                lambda_smooth=None,
                                lambda_action=None,
                                lambda_accer=None,
                                lambda_steer=None,
                                lambda_img=None,
                                regular_action_loss=None,
                                obs_r_length=None,
                                input_types=None,
                                loss_type=None,
                                batch_size=None,
                                save_steps=None,
                                is_multi_frames=None,
                                is_multi_text_obs=None,
                                horizon=None,
                                writer=None,
                                is_change_text_feature=False,
                                micro_batch_size=None,
                                global_step=None,
                                input_real_seq_length=None,
                                llm_real_labels=None,
                                shift_labels=False,
                                is_truncation_logits=False,
                                is_reconstruction=False,
                                orignal_img=None,
                                img_reconstruction=None):
        global action_loss, loss, logits, smoothed_loss, save_train_step, log_name, img_loss
        epsilon = 0.1
        ignore_index = -100
        save_train_step = 2
        if input_types == ['obs', 'text'] or input_types == ['text']:

            if len(input_types) == 1:
                if is_change_text_feature:
                    logits = predict_text_logits
                    log_name = ['get logits from custom_lm_head']
                else:
                    logits = model_outputs["logits"]
                    log_name = ['get logits from default_lm_head']

                labels = llm_real_labels

            else:
                if is_change_text_feature:
                    logits = predict_text_logits
                    log_name = ['get logits from custom_lm_head']
                elif is_truncation_logits:
                    logits = model_outputs["logits"][:, obs_r_length:]
                    log_name = ['get logits from default_lm_head']
                else:
                    logits = model_outputs["logits"]
                    log_name = ['get logits from default_lm_head']


                if llm_real_labels is None:
                    padded_labels = -100 * torch.ones(logits.shape[0], logits.shape[1])

                    for i in range(logits.shape[0]):
                        seq_length = input_real_seq_length[i]
                        if seq_length + 1 < labels.shape[1] and seq_length + 1 < padded_labels.shape[1]:
                            padded_labels[i, :seq_length + 1] = labels[i, :seq_length + 1]

                    labels = padded_labels.type(torch.int64)
                elif is_truncation_logits:
                    labels = llm_real_labels
                else:
                    labels = torch.nn.functional.pad(llm_real_labels, (padding_left, 0), value=-100)

            cutoff_length1 = self.model.cutoff_length
            cutoff_length2 = cutoff_length1 + 1

            if shift_labels:
                logits = logits[..., -cutoff_length2:-2, :].contiguous() # default=424+1, only multi_frames16
            else:
                logits = logits[..., -cutoff_length1:, :].contiguous()
            if labels.shape[0] % horizon == 0:
                if is_multi_frames:
                    labels = labels.reshape(labels.shape[0] // horizon, horizon, -1)[:, -1].unsqueeze(-1)
                    action_labels = action_labels.reshape(action_labels.shape[0] // horizon, horizon, -1)[:, -1]
                    if shift_labels:
                        labels = labels[..., 1:, :].contiguous()
                else:
                    labels = labels.reshape(labels.shape[0] // horizon, -1).unsqueeze(-1)
                    action_labels = action_labels.reshape(action_labels.shape[0] // horizon, horizon, -1)[:, -1]
            else:
                labels = labels[-1].unsqueeze(0)
                if shift_labels:
                    labels = labels[..., 1:].contiguous().unsqueeze(-1)
                action_labels = action_labels[-1].unsqueeze(0)

            log_probs = -nn.functional.log_softmax(logits, dim=-1).to('cuda')
            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)
            padding_mask = labels.eq(ignore_index).to('cuda')
            # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
            # will ignore them in any case.
            labels = torch.clamp(labels, min=0)
            nll_loss = log_probs.gather(dim=-1, index=labels.to('cuda'))
            # works for fp16 input tensor too, by internally upcasting it to fp32
            smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
            nll_loss.masked_fill_(padding_mask, 0.0)
            smoothed_loss.masked_fill_(padding_mask, 0.0)
            # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            nll_loss = nll_loss.sum() / num_active_elements
            smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
            smoothed_loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss
            if loss_type == ['normal', 'mean'] or loss_type == ['normal']:
                actual_actions_normalized = torch.clone(action_labels)
                actual_actions_normalized[:, 0] = (action_labels[:, 0] + 3) / 6
                actual_actions_normalized[:, 1] = (action_labels[:, 1] + 0.2) / 0.4
                predicted_actions_normalized = torch.clone(actor_predict_actions)
                predicted_actions_normalized[:, 0] = (actor_predict_actions[:, 0] + 3) / 6
                predicted_actions_normalized[:, 1] = (actor_predict_actions[:, 1] + 0.2) / 0.4
                if len(loss_type) == 2:
                    action_loss = F.mse_loss(predicted_actions_normalized, actual_actions_normalized, reduction='none').mean(dim=(0, 1))
                else:
                    action_loss = F.mse_loss(predicted_actions_normalized, actual_actions_normalized)
            elif loss_type == ['mean']:
                action_loss = F.mse_loss(actor_predict_actions, action_labels, reduction='none').mean(dim=(0, 1))
            elif loss_type == ['raw']:
                action_loss = F.mse_loss(actor_predict_actions, action_labels)
            else:
                raise ValueError('Please give a type of loss')
            if regular_action_loss and loss_type == ['raw']:
                a_values = actor_predict_actions[:, 0]
                s_values = actor_predict_actions[:, 1]
                accer_penalty = torch.mean(torch.clamp(a_values - 3, min=0) ** 2 + torch.clamp(-3 - a_values, min=0) ** 2)
                steer_penalty = torch.mean(torch.clamp(s_values - 0.25, min=0) ** 2 + torch.clamp(-0.25 - s_values, min=0) ** 2)
                loss = lambda_smooth * smoothed_loss + lambda_action * action_loss + lambda_accer * accer_penalty + lambda_steer * steer_penalty
            else:
                loss = lambda_smooth * smoothed_loss + lambda_action * action_loss
            # update decision model via mse loss
            if len(input_types) == 2:
                if self.state.global_step % save_train_step == 0:
                    writer.add_scalar(f'Training Smoothed_loss/{log_name[0]} ', lambda_smooth * smoothed_loss,
                                      self.state.global_step)
                    writer.add_scalar(f'Training MSE_loss/{log_name[0]} ', action_loss, self.state.global_step)
                    writer.add_scalar(f'Training Obs_Text_loss/{log_name[0]} ', loss, self.state.global_step)

                if self.state.global_step % save_steps == 0:
                    writer.add_scalar(f'Evaluate Smoothed_loss/{log_name[0]}  ', lambda_smooth * smoothed_loss,
                                      self.state.global_step)
                    writer.add_scalar(f'Evaluate MSE_loss/{log_name[0]}  ', action_loss, self.state.global_step)
                    writer.add_scalar(f'Evaluate Obs_Text_loss/{log_name[0]}  ', loss, self.state.global_step)
                print('Step_Obs_Text_loss: ', loss)
            else:
                if self.state.global_step % save_train_step == 0:
                    writer.add_scalar(f'Training Smoothed_loss/{log_name[0]}  ', lambda_smooth * smoothed_loss,
                                      self.state.global_step)
                    writer.add_scalar(f'Training MSE_loss/{log_name[0]}  ', action_loss, self.state.global_step)
                    writer.add_scalar(f'Training Text_loss/{log_name[0]}  ', loss, self.state.global_step)

                if self.state.global_step % save_steps == 0:
                    writer.add_scalar(f'Evaluate Smoothed_loss/{log_name[0]}  ', lambda_smooth * smoothed_loss,
                                      self.state.global_step)
                    writer.add_scalar(f'Evaluate MSE_loss/{log_name[0]}  ', action_loss, self.state.global_step)
                    writer.add_scalar(f'Evaluate Text_loss/{log_name[0]}  ', loss, self.state.global_step)
                print('Step_only_Text_loss: ', loss)

        elif input_types == ['obs']:
            # update decision model via mse loss
            if labels.shape[0] % horizon == 0:
                if not is_multi_text_obs:
                    labels = labels.reshape(labels.shape[0] // horizon, horizon, -1)[:, -1].unsqueeze(-1)
                    action_labels = action_labels.reshape(action_labels.shape[0] // horizon, horizon, -1)[:, -1]
                else:
                    labels = labels.reshape(labels.shape[0] // horizon, -1).unsqueeze(-1)
                    action_labels = action_labels.reshape(action_labels.shape[0] // horizon, horizon, -1)[:, -1]
            action_loss = F.mse_loss(actor_predict_actions, action_labels)
            loss = lambda_action * action_loss
            print('MSE_loss', action_loss)
            print('Action_loss', lambda_action * action_loss)
            print('Step_only_obs_loss: ', loss)
            if self.state.global_step % save_train_step == 0:
                writer.add_scalar('Training MSE_loss ', self.state.global_step)
                writer.add_scalar('Training Obs_loss ', self.state.global_step)

            if self.state.global_step % save_steps == 0:
                writer.add_scalar('Evaluate MSE_loss ', action_loss, self.state.global_step)
                writer.add_scalar('Evaluate Obs_loss ', loss, self.state.global_step)
        else:
            raise TypeError(f"Input types is not in input_list")

        if input_types == ['obs'] or input_types == ['obs', 'text']:
            if self.model.is_reconstruction or self.model.is_multi_reconstruct:
                img_loss = F.mse_loss(orignal_img, img_reconstruction)
                print(f'img_loss: {img_loss}')
                if self.state.global_step % save_train_step == 0:
                    writer.add_scalar(f'Training {input_types}_loss/{log_name[0]} ', img_loss, self.state.global_step)

                if self.state.global_step % save_steps == 0:
                    writer.add_scalar(f'Evaluate {input_types}_loss/{log_name[0]} ', img_loss, self.state.global_step)
                loss = loss + lambda_img * img_loss
                print(f'Step_all_img_loss: {loss}')
        if input_types == ["obs"]:
            if torch.any(torch.isnan(action_loss)):
                loss = torch.tensor(4.0, requires_grad=True).to('cuda')
        else:
            if torch.any(torch.isnan(smoothed_loss)) or torch.any(torch.isnan(action_loss)):
                loss = torch.tensor(4.0, requires_grad=True).to('cuda')
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        llm_labels = inputs["labels"]
        llm_real_labels = inputs["real_labels"]
        input_real_seq_length = inputs["input_real_seq_length"]
        action_labels = inputs["action"]
        sum_rewards_labels = inputs["sum_rewards"]
        inputs_obs = inputs["inputs_obs"]
        inputs_llm_mask = inputs['attention_mask']
        inputs_llm_ids = inputs['input_ids']

        if self.model.is_embeds_clip:
            image_embeds_clip = inputs["image_embeds_clip"]
            text_embeds_clip = inputs["text_embeds_clip"]
        else:
            image_embeds_clip = None
            text_embeds_clip = None
        llm_length, model_outputs, actor_predict_actions, predict_text_logits,\
        input_ids_from_embeds,  obs_r_length, img_reconstruction = model(llm_labels,
                                                    inputs_llm_ids,
                                                    input_real_seq_length,
                                                    inputs_llm_mask,
                                                    inputs_obs,
                                                    action_labels,
                                                    sum_rewards_labels,
                                                    input_real_seq_length,
                                                    llm_real_labels,
                                                    image_embeds_clip,
                                                    text_embeds_clip)

        loss = self.label_smoother_mse_loss(
                        model_outputs=model_outputs,
                        labels=llm_labels,
                        action_labels=action_labels,
                        actor_predict_actions=actor_predict_actions,
                        predict_text_logits=predict_text_logits,
                        lambda_smooth=self.model.lambda_smooth,
                        lambda_action=self.model.lambda_action,
                        lambda_accer=self.model.lambda_accer,
                        lambda_steer=self.model.lambda_steer,
                        lambda_img=self.model.lambda_img,
                        regular_action_loss=self.model.regular_action_loss,
                        obs_r_length=obs_r_length,
                        input_types=self.model.input_types,
                        loss_type=self.model.loss_type,
                        batch_size=self.model.batch_size,
                        save_steps=self.model.save_steps,
                        writer=self.model.writer,
                        is_multi_frames=self.model.is_multi_frames,
                        is_multi_text_obs=self.model.is_multi_text_obs,
                        horizon=self.model.horizon,
                        shift_labels=self.model.shift_labels,
                        is_change_text_feature=self.model.is_change_text_feature,
                        micro_batch_size=self.model.micro_batch_size,
                        global_step=self.model.global_step,
                        input_real_seq_length=input_real_seq_length,
                        llm_real_labels=llm_real_labels,
                        is_truncation_logits=self.model.is_truncation_logits,
                        is_reconstruction=self.model.is_reconstruction,
                        orignal_img=inputs_obs,
                        img_reconstruction=img_reconstruction,)

        return (loss, model_outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model, tokenizer, and the args into a sub-folder under `output_dir`.
        If `self.is_world_process_zero()` is True, then this method saves a checkpoint to `output_dir`.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            raise ValueError("Model does not have a 'save_pretrained' method.")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Load the model from a checkpoint.
        """
        if not resume_from_checkpoint:
            return
        self.state.best_model_checkpoint = None
        self.model = VlmDTModel.from_pretrained(resume_from_checkpoint, self.model.lora_model, self.tokenizer)

    def _issue_warnings_after_load(self, load_result):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        for key, value in inputs.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise ValueError(f"Input {key} contains NaN or Inf values.")

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.args.max_grad_norm = 1000 # default = 1.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"Gradient for {name} contains NaN or Inf values.")
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach() / self.args.gradient_accumulation_steps


class CustomDataset_pre(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


statistic_input_ids_length = []


class CLIPEmbeddingExtractor:
    def __init__(self, model_name='openai/clip-vit-base-patch16', max_length=77, num_heads=16):
        self.config = CLIPConfig.from_pretrained(model_name)
        self.config.text_config.max_position_embeddings = max_length
        self.config.text_config.hidden_size = 4096
        self.config.vision_config.hidden_size = 4096
        self.config.projection_dim = 4096
        self.config.vision_config.num_attention_heads = num_heads
        self.config.text_config.num_attention_heads = num_heads
        self.max_length = max_length
        self.clip_model = CLIPModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def preprocess_batch(self, batch_images):
        batch_images_pil = [transforms.ToPILImage()(img) for img in batch_images]
        processed_images = torch.stack([self.transform(img) for img in batch_images_pil])
        return processed_images

    def get_embeddings(self, images, text):
        processed_images = self.preprocess_batch(images)
        inputs = self.clip_processor(images=processed_images, text=text,
                                     return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = self.clip_model(pixel_values=pixel_values, input_ids=input_ids)
        clip_image_last_hidden_state = outputs.vision_model_output.last_hidden_state
        clip_text_last_hidden_state = outputs.text_model_output.last_hidden_state
        padding_size = self.max_length - clip_text_last_hidden_state.shape[1]
        clip_text_last_hidden_state = F.pad(clip_text_last_hidden_state, (0, 0, 0, padding_size))

        return clip_image_last_hidden_state.squeeze(0), clip_text_last_hidden_state.squeeze(0)

    def decode_text(self, input_ids):
        texts = self.clip_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return texts

class CustomDataset(nn.Module):

    def __init__(self,
                 file_path,
                 tokenizer,
                 prompter,
                 input_types,
                 cutoff_length,
                 train_on_inputs,
                 clip_extractor,
                 is_embeds_clip,
                 is_big_buffer,
                 is_step_reward,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.input_types = input_types
        self.prompter = prompter
        self.train_on_inputs = train_on_inputs
        self.is_embeds_clip = is_embeds_clip
        self.clip_extractor = clip_extractor
        self.is_big_buffer = is_big_buffer
        self.is_step_reward = is_step_reward
        if self.input_types == ['obs', 'text'] or ['text']:
            self.cutoff_length = cutoff_length
        else:
            self.cutoff_length = cutoff_length - 120


    def tokenize_no_padding(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=False,
            max_length=self.cutoff_length,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_length
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(self, data_point, add_eos_token=True):
        full_prompt = self.prompter.generate_prompt(
            data_point["input"],
            data_point["Question"],
            data_point["Answer"]
        )
        tokenized_full_prompt = self.tokenize_no_padding(full_prompt)
        input_real_seq_length = len(tokenized_full_prompt["input_ids"])
        input_real_seq_length_padding = [0] * (self.cutoff_length - input_real_seq_length)
        tokenized_full_prompt["input_ids"].extend(input_real_seq_length_padding)
        input_real_seq_length_padding_mask = [0] * (self.cutoff_length - input_real_seq_length - 1)
        tokenized_full_prompt["attention_mask"].append(1)
        tokenized_full_prompt["input_real_seq_length"] = input_real_seq_length
        tokenized_full_prompt["attention_mask"].extend(input_real_seq_length_padding_mask)

        if self.is_big_buffer:
            tokenized_full_prompt["inputs_obs"] = data_point["lidar_noground"]["lidar_noground"]
            tokenized_full_prompt["action"] = data_point["action"][0]
        else:
            tokenized_full_prompt["inputs_obs"] = data_point["lidar_noground"] #　training
            # tokenized_full_prompt["inputs_obs"] = data_point["lidar_noground"]["lidar_noground"] # evaluating
            tokenized_full_prompt["action"] = data_point["action"]
        tokenized_full_prompt["ones"] = 1
        if not self.is_step_reward:
            if type(data_point["reward"]) is not list:
                tokenized_full_prompt["sum_rewards"] = [data_point["reward"]]
            else:
                tokenized_full_prompt["sum_rewards"] = data_point["reward"]
        else:
            tokenized_full_prompt["sum_rewards"] = data_point["sum_rewards"]

        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        tokenized_full_prompt["real_labels"] = tokenized_full_prompt["input_ids"][:input_real_seq_length + 1] + [
            -100] * len(input_real_seq_length_padding_mask)

        if len(tokenized_full_prompt["input_ids"]) - (
                self.cutoff_length - input_real_seq_length - 1) > self.cutoff_length - 264:
            statistic_input_ids_length.append(len(tokenized_full_prompt["input_ids"]) - (self.cutoff_length - input_real_seq_length - 1))


        if self.is_embeds_clip:
            batch_images = torch.tensor(data_point["lidar_noground"])
            text = full_prompt
            image_embeddings, text_embeddings = self.clip_extractor.get_embeddings(batch_images, text)

            tokenized_full_prompt["image_embeds_clip"] = image_embeddings
            tokenized_full_prompt["text_embeds_clip"] = text_embeddings
            inputs = self.clip_extractor.clip_processor(text=text, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=self.cutoff_length)
            input_ids = inputs['input_ids']

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["input"], data_point["Question"]
            )
            tokenized_user_prompt = self.tokenize_no_padding(
                user_prompt, add_eos_token=add_eos_token
            )
            print(self.tokenizer.decode(tokenized_user_prompt["input_ids"], skip_special_tokens=False))
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 2
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["real_labels"][
                                                                         user_prompt_len:]
            test_labels = tokenized_full_prompt["labels"]
            test_labels = [0 if x < 0 else x for x in test_labels]
            tokenized_full_prompt["real_labels"] = tokenized_full_prompt["labels"].copy()
            # could be sped up, probably
        return tokenized_full_prompt


def train(
    # model/data params
    base_model: str = "",   # base model name/path (required)
    data_path: str = "",    # dataset path
    data_path_pkl: str = "",   # optional pickle dataset path
    output_dir: str = "./MIMO_VLA_training_results",   # output directory for checkpoints/logs
    prompt_template: str = "templates/alpaca",   # template for input prompts

    # training hyperparams
    batch_size: int = 128,       # batch size
    micro_batch_size: int = 4,   # micro-batch size for gradient accumulation
    num_epochs: int = 3,         # number of training epochs
    learning_rate: float = 3e-4, # learning rate
    cutoff_len: int = 424,       # max sequence length (default=256)
    val_set_size: int = 100,     # validation set size
    save_step: int = 10,         # save checkpoint every N steps

    # lora hyperparams
    lora_r: int = 8,             # LoRA rank
    lora_alpha: int = 16,        # LoRA alpha scaling
    lora_dropout: float = 0.05,  # LoRA dropout probability
    lora_target_modules: List[str] = [   # modules to apply LoRA
        "q_proj",
        "k_proj",
        "o_proj",
        "v_proj"
    ],

    # llm hyperparams
    train_on_inputs: bool = False,   # if False, mask out inputs in loss
    add_eos_token: bool = False,     # whether to add EOS token
    group_by_length: bool = True,    # group sequences by length (faster, uneven loss curve)
    length_column_name = "input",    # column name for length in dataset

    # wandb params
    wandb_project: str = "",         # W&B project name
    wandb_run_name: str = "",        # W&B run name
    wandb_watch: str = "",           # W&B watch mode: false | gradients | all
    wandb_log_model: str = "",       # W&B log model: false | true
    resume_from_checkpoint: str = "",   # resume training from checkpoint path
    # resume_from_checkpoint: str = ".../checkpoint-850",  # example

    prompt_template_name: str = "templates/alpaca",   # path to prompt template

    # task/custom params
    input_types: List[str] = ["obs", "text"],   # input modalities (e.g., obs, text)
    loss_type: List[str] = ["normal", "mean"],  # loss function types
    is_change_text_feature = False,             # whether to modify text features
    default_input_embedding_layer = True,       # use default input embedding layer
    is_attention_fusion: bool = False,          # enable attention fusion between inputs
    fc_fusion_layers: int = 0,                  # number of FC fusion layers
    is_embeds_clip: bool = False,               # clip embeddings if True
    is_truncation_logits: bool = False,         # truncate logits if True
    shift_labels: bool = False,                 # shift labels for loss if True

    # loss weights
    lambda_action: int = 1,   # action loss weight
    lambda_smooth: int = 1,   # smoothness loss weight
    lambda_accer: int = 1,    # acceleration loss weight
    lambda_steer: int = 1,    # steering loss weight
    lambda_img: int = 1,      # image reconstruction loss weight

    # training/task settings
    is_multi_frames: bool = False,       # use multiple frames as input
    is_multi_text_obs: bool = False,     # use multiple text observations
    horizon: int = 1,                    # prediction horizon
    is_big_buffer: bool = False,         # enable large buffer mode
    is_redirect: bool = True,            # redirect logs/outputs
    regular_action_loss: bool = False,   # use regularized action loss
    img_patch_size: int = 32,            # patch size for image embedding
    is_step_reward: bool = False,        # use step-wise rewards
    is_reconstruction: bool = False,     # enable image reconstruction
    is_multi_reconstruct: bool = False,  # reconstruct multiple inputs

):

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    os.makedirs(output_dir, exist_ok=True)
    output_log_path = os.path.join(output_dir, f'output_log_{output_dir}.txt')
    if is_redirect:
        f = open(output_log_path, 'w')
        sys.stdout = f
        sys.stderr = f
    try:
        global train_data, val_data
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Training Alpaca-LoRA model with params:\n"
                f"base_model: {base_model}\n"
                f"data_path: {data_path}\n"
                f"data_path_pkl: {data_path_pkl}\n"
                f"output_dir: {output_dir}\n"
                f"prompt_template: {prompt_template}\n"
                f"batch_size: {batch_size}\n"
                f"micro_batch_size: {micro_batch_size}\n"
                f"num_epochs: {num_epochs}\n"
                f"learning_rate: {learning_rate}\n"
                f"cutoff_len: {cutoff_len}\n"
                f"val_set_size: {val_set_size}\n"
                f"save_step: {save_step}\n"
                f"lora_r: {lora_r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"lora_dropout: {lora_dropout}\n"
                f"lora_target_modules: {lora_target_modules}\n"
                f"train_on_inputs: {train_on_inputs}\n"
                f"add_eos_token: {add_eos_token}\n"
                f"group_by_length: {group_by_length}\n"
                f"wandb_project: {wandb_project}\n"
                f"wandb_run_name: {wandb_run_name}\n"
                f"wandb_watch: {wandb_watch}\n"
                f"loss_type: {loss_type}\n"
                f"wandb_log_model: {wandb_log_model}\n"
                f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                f"prompt template: {prompt_template_name}\n"
                f"input_types: {input_types}\n"
                f"is_change_text_feature: {is_change_text_feature}\n"
                f"default_input_embedding_layer: {default_input_embedding_layer}\n",
                f"is_attention_fusion: {is_attention_fusion}\n",
                f"fc_fusion_layers: {fc_fusion_layers}\n",
                f"is_embeds_clip: {is_embeds_clip}\n",
                f"is_truncation_logits: {is_truncation_logits}\n",
                f"lambda_action: {lambda_action}\n",
                f"lambda_smooth: {lambda_smooth}\n",
                f"lambda_accer: {lambda_accer}\n",
                f"lambda_steer: {lambda_steer}\n",
                f"lambda_img: {lambda_img}\n",
                f"is_multi_frames: {is_multi_frames}\n",
                f"is_multi_text_obs: {is_multi_text_obs}\n",
                f"horizon {horizon}\n",
                f"is_big_buffer: {is_big_buffer}\n"
                f"is_redirect: {is_redirect}\n",
                f"regular_action_loss: {regular_action_loss}\n",
                f"img_patch_size: {img_patch_size}\n",
                f"is_step_reward: {is_step_reward}\n",
                f"is_reconstruction: {is_reconstruction}\n",
                f"shift_labels: {shift_labels}\n"
                f"is_multi_reconstruct: {is_multi_reconstruct}",

            )
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
        gradient_accumulation_steps = batch_size // micro_batch_size
        prompter = Prompter(prompt_template_name)
        device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            gradient_accumulation_steps = gradient_accumulation_steps // world_size
        # wandb.init(reinit=True)
        # wandb.login(relogin=True)
        os.environ["WANDB_MODE"] = "offline"
        # Check if parameter passed or if set within environ
        use_wandb = len(wandb_project) > 0 or (
                "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
        # Only overwrite environ if wandb param passed
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
        assert isinstance(model, object)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        # tokenizer.add_special_tokens()

        model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        model = get_peft_model(model, config)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        dataset = CustomDataset_pre(data)
        if is_embeds_clip:
            clip_extractor = CLIPEmbeddingExtractor(max_length=cutoff_len)
        else:
            clip_extractor = None
        datasets_map = CustomDataset(data_path,
                                     tokenizer,
                                     prompter,
                                     input_types,
                                     cutoff_len,
                                     train_on_inputs,
                                     clip_extractor,
                                     is_embeds_clip,
                                     is_big_buffer,
                                     is_step_reward)
        if val_set_size < 1:
            train_size = int((1 - val_set_size) * len(dataset))
            test_size = len(dataset) - train_size
        else:
            test_size = val_set_size
            train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = HFDataset.from_list(train_dataset)
        train_data = train_dataset.map(datasets_map.generate_and_tokenize_prompt)
        test_dataset = [test_dataset[i] for i in range(len(test_dataset))]
        test_dataset = HFDataset.from_list(test_dataset)
        val_data = test_dataset.map(datasets_map.generate_and_tokenize_prompt)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        output_log = os.path.join(f"{output_dir}/training_{output_dir}_logs")
        os.makedirs(output_log, exist_ok=True)
        writer = SummaryWriter(output_log)

        vlm_dtmodel = VlmDTModel(original_model=model,
                                 input_types=input_types,
                                 loss_type=loss_type,
                                 save_steps=save_step,
                                 batch_size=batch_size,
                                 img_patch_size=img_patch_size,
                                 lora_config=config,
                                 lora_target_modules=lora_target_modules,
                                 writer=writer,
                                 is_change_text_feature=is_change_text_feature,
                                 micro_batch_size=micro_batch_size,
                                 default_input_embedding_layer=default_input_embedding_layer,
                                 is_atten_fusion=is_attention_fusion,
                                 fc_fusion_layers=fc_fusion_layers,
                                 is_embeds_clip=is_embeds_clip,
                                 shift_labels=shift_labels,
                                 tokenizer=tokenizer,
                                 lambda_action=lambda_action,
                                 lambda_smooth=lambda_smooth,
                                 lambda_steer=lambda_steer,
                                 lambda_accer=lambda_accer,
                                 lambda_img=lambda_img,
                                 horizon=horizon,
                                 cutoff_length=cutoff_len,
                                 is_multi_frames=is_multi_frames,
                                 is_multi_text_obs=is_multi_text_obs,
                                 regular_action_loss=regular_action_loss,
                                 is_truncation_logits=is_truncation_logits,
                                 is_reconstruction=is_reconstruction,
                                 is_multi_reconstruct=is_multi_reconstruct)


        if not ddp and torch.cuda.device_count() > 1:
            # vlm_dtmodel = nn.DataParallel(vlm_dtmodel)
            vlm_dtmodel.lora_model.is_parallelizable = True
            vlm_dtmodel.lora_model.model_parallel = True
        print('starting training')


        trainer = CustomTrainer(model=vlm_dtmodel,
                                train_dataset=train_data,
                                eval_dataset=val_data,
                                args=transformers.TrainingArguments(
                                    per_device_train_batch_size=micro_batch_size,
                                    gradient_accumulation_steps=gradient_accumulation_steps,
                                    warmup_steps=100,
                                    num_train_epochs=num_epochs,
                                    learning_rate=learning_rate,
                                    fp16=False,
                                    logging_steps=10,
                                    optim="adamw_torch",
                                    evaluation_strategy="steps" if val_set_size > 0 else "no",
                                    save_strategy="steps",
                                    eval_steps=25 if val_set_size > 0 else None,
                                    save_steps=25,
                                    output_dir=output_dir,
                                    save_total_limit=3,
                                    load_best_model_at_end=True if val_set_size > 0 else False,
                                    ddp_find_unused_parameters=False if ddp else None,
                                    group_by_length=group_by_length,
                                    report_to="wandb" if use_wandb else None,
                                    run_name=wandb_run_name if use_wandb else None,
                                ),
                                data_collator=transformers.DataCollatorForSeq2Seq(
                                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
                                tokenizer=tokenizer,
                                )
        trainer.state.best_model_checkpoint = None
        trainer.model.lora_model.config.use_cache = False
        # Disable gradient checkpointing
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     vlm_dtmodel.lora_model = torch.compile(vlm_dtmodel.lora_model)sts
        # must add this line on V100 32G
        old_state_dict = vlm_dtmodel.lora_model.state_dict
        vlm_dtmodel.lora_model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(vlm_dtmodel.lora_model, type(vlm_dtmodel.lora_model))
        with torch.autocast("cuda"):
            if resume_from_checkpoint:
                print('load checkpoint')
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                print('no checkpoint')
                trainer.train()
        # vlm_dtmodel.lora_model.save_pretrained(output_dir)
        vlm_dtmodel.save_pretrained(output_dir)
        writer.close()
        # trainer.save_model(output_dir2)
        print('Successfully training')

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if is_redirect:
            f.close()


if __name__ == "__main__":
    fire.Fire(train)

