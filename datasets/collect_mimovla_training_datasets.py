import pickle
import json
import torch
import re
import torch.nn as nn
import numpy as np
import random
import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
# from memory import ExperienceReplay
data_path = 'EGADS_Agent_all_datasets_infos.pkl'
datasets = []
patch_size = 16
embed_dim = 128
class ImageEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super(ImageEmbedding, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
    def forward(self, data_dict):
        x = self.proj(data_dict)  # (B, embed_dim, num_patches_w, num_patches_h)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        # add position embedding
        x = x + self.position_embedding
        print(x.shape)
        x = x.squeeze(dim=0)
        x = x.flatten(0)
        print(x.shape)
        return x
imageSplit_model = ImageEmbedding(128, patch_size, embed_dim)

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

if torch.cuda.is_available():
  print("using CUDA")
  device = torch.device('cuda')
else:
  print("using CPU")
  device = torch.device('cpu')
collect_egads = False
input_names = ['lidar_noground']


def replace_action_format(s):
    import re
    match = re.search(r"action=\[([\d\.]+),\s*([\d\.]+)\]", s)
    if match:
        return "action=[action1, action2]"
    return s

with open('MIMO_VLA_training_datasets.pkl', 'wb') as f:
    with open(data_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        text_eg1 = "You can see that there is 1 car. It is speed, straight-line distance from you, and angle in the direction your heading are respectively  0.0 m/s,  14.84 m,  124.43 ° . You are now -0.000m laterally away from your driving route."
        text_eg2 = "You can see that there are 3 cars. Their speed, straight-line distance from you, and angle in the direction your heading are respectively  0.0 2 6 m/s,  14.84 34 54 m,  124.43 323 656 °. You are now -0.000m laterally away from your driving route."
        text_eg3 = "You see no car here, and you are now 43 m laterally away from your driving route."
        pattern_eg1 = re.compile(
            r"there is (\d+) car\. It is speed, straight-line distance from you, and angle in the direction your heading are respectively\s*([\d\.]+)\s*m/s,\s*([\d\.]+)\s*m,\s*([\d\.]+)\s*°\s*\. You are now\s*([\-\d\.]+)m laterally away from your driving route\."
        )

        pattern_eg2 = re.compile(
            r"there are (\d+) cars, their speed, their straight-line distance from you, and their Angle in the direction your heading are \s*([\d\s\.]+)m/s,\s*([\d\s\.]+)m,\s*([\d\s\.]+)°\s*\. You are now\s*([\-\d\.]+)m laterally away from your driving route\."
        )

        pattern_eg3 = re.compile(
            r"You see no car here, and you are now\s*([\-\d\.]+)m laterally away from your driving route\."
        )
        def parse_example(text):
            if "there is" in text:
                match = pattern_eg1.search(text)
                if match:
                    car_count = int(match.group(1))
                    speeds = [float(match.group(2))]
                    distances = [float(match.group(3))]
                    angles = [float(match.group(4))]
                    lateral = float(match.group(5))
                    return {"car_count": car_count, "speeds": speeds, "distances": distances, "angles": angles,
                            "lateral": lateral}
            elif "there are" in text:
                match = pattern_eg2.search(text)
                if match:
                    car_count = int(match.group(1))
                    speeds = list(map(float, match.group(2).strip().split()))
                    distances = list(map(float, match.group(3).strip().split()))
                    angles = list(map(float, match.group(4).strip().split()))
                    lateral = float(match.group(5))
                    return {"car_count": car_count, "speeds": speeds, "distances": distances, "angles": angles,
                            "lateral": lateral}
            elif "You see no car" in text:
                match = pattern_eg3.search(text)
                if match:
                    lateral = float(match.group(1))
                    return {"car_count": 0,  "speeds": None, "distances": None, "angles": None,  "lateral": lateral}
            return None
        pattern_output = re.compile(
            r"Because your current yaw anngle and speed are\s*([-\d\.]+)\u00b0,\s*([-\d\.]+)m/s"
        )

        def replace_action_tokens(text, action):
            text = text.replace("{}".format(action[0]), "<action0>")
            text = text.replace("{}".format(action[1]), "<action1>")
            return text

        for i in range(len(data)):
            print('#####################{}###############'.format(i))
            instruction = data[i]["instruction"]
            original_input = data[i]["input"]
            print(original_input)
            result_input = parse_example(original_input)
            print('get_result', result_input)
            output_text = data[i]["output"]
            match = pattern_output.search(output_text)
            if match:
                yaw_angle = float(match.group(1))
                speed = float(match.group(2))
                extracted_data = {"yaw_angle": yaw_angle, "speed": speed}
            else:
                extracted_data = {"yaw_angle": None, "speed": None}
            print("Extracted Data:", extracted_data)
            if collect_egads:
                lidar_ng = data[i]["observation"]
                action = data[i]["actions"]
                reward = data[i]["reward"]
            else:
                lidar_ng = data[i]["observation"]["lidar_noground"].tolist()
                action = data[i]["actions"].tolist()
                reward = data[i]["reward"].tolist()

            none_str = ""
            done = False if instruction == none_str else True
            data[i]["done"] = done

            sum_rewards = data[i]["sum_rewards"].tolist()
            chosen_question = random.choice(questions)
            print('chose_question', chosen_question)

            ego_angle, ego_speed = extracted_data['yaw_angle'], extracted_data['speed'],
            vehicles_num, multi_dis, multi_yaw, speed_round, lateral_dis_round = result_input['car_count'], result_input['distances'], result_input['angles'], result_input['speeds'], result_input['lateral']
            if chosen_question == "What are you seeing/observing?" and vehicles_num == 0:
                answer = "Sorry, I didn’t see any vehicles."

            elif chosen_question == "What are you seeing/observing?" and vehicles_num == 1:
                answer = "I am observing {} car. It is at a straight-line distance of {} meters from me positioned at an angle of {}° relative to my heading.".format(
                    vehicles_num, multi_dis, multi_yaw)
            elif chosen_question == "What are you seeing/observing?" and vehicles_num > 1:
                answer = "I am observing {} cars. They are respectively at straight-line distance of {} meters from me positioned at an angle of {}° relative to my heading.".format(
                    vehicles_num, multi_dis, multi_yaw)

            elif chosen_question == "What are you paying attention to and why?" and vehicles_num == 0:
                answer = "Although there are no vehicles currently present, I need to constantly monitor the surrounding environment for safer driving."
            elif chosen_question == "What are you paying attention to and why?" and vehicles_num == 1:
                answer = "I am paying attention to the car {} meters away at an angle of {}° because it may impact my next driving decision.".format(
                    multi_dis, multi_yaw)
            elif chosen_question == "What are you paying attention to and why?" and vehicles_num > 1:
                answer = "I am paying attention to the cars {} meters away at angle of {}° because it may impact my next driving decision.".format(
                    multi_dis, multi_yaw)

            elif chosen_question == "Are there any traffic lights? What’s the color of the traffic light?":
                answer = "There are no traffic lights mentioned in the input."
            elif chosen_question == "What’s your current speed and steering angle?":
                answer = "My current speed is {} m/s and steering angle is {} m/s.".format(ego_speed, ego_angle)
            elif chosen_question == "What is your action and why?":
                answer = "My action is to proceed safely with action=[{}, {}] to maintain safe driving given the car's position and my current state.".format(
                    action[0], action[1])

            elif chosen_question == "Summarize the current driving scenario at a high level." and vehicles_num == 0:
                answer = "Currently, I am stationary with {} car.".format(
                    vehicles_num)
            elif chosen_question == "Summarize the current driving scenario at a high level." and vehicles_num == 1:
                answer = "Currently, I am stationary with {} car detected {} meters away at a {}° angle. I need to plan a safe maneuver to proceed.".format(
                    vehicles_num, multi_dis, multi_yaw)
            elif chosen_question == "Summarize the current driving scenario at a high level." and vehicles_num > 1:
                answer = "Currently, I am stationary with {} cars detected {} meters away at {}° angle. I need to plan a safe maneuver to proceed.".format(
                    vehicles_num, multi_dis, multi_yaw)

            elif chosen_question == "How are you going to drive in this situation and why?":
                answer = "I will proceed with a slight steering adjustment and an acceleration to ensure I maintain a safe path relative to the detected car(s)."
            elif chosen_question == "What’s the straight-line distance to the nearest car?" and vehicles_num == 0:
                answer = "Sorry, I didn’t see any vehicles."
            elif chosen_question == "What’s the straight-line distance to the nearest car?" and vehicles_num == 1:
                answer = "The straight-line distance to the nearest car is {} meters.".format(multi_dis)
            elif chosen_question == "What’s the straight-line distance to the nearest car?" and vehicles_num > 1:
                answer = "The straight-line distance to the nearest car is {} meters.".format(multi_dis)

            elif chosen_question == "What is the angle of the nearest car relative to your heading?" and  vehicles_num == 0:
                answer = "Sorry, I didn’t see any vehicles."

            elif chosen_question == "What is the angle of the nearest car relative to your heading?" and vehicles_num == 1:
                answer = "The angle of the nearest car relative to my heading is {}°.".format(multi_yaw)

            elif chosen_question == "What is the angle of the nearest car relative to your heading?" and vehicles_num >1 :
                answer = "The angle of the nearest car relative to my heading are respectively {}°.".format(multi_yaw)


            elif chosen_question == "Is there any lateral deviation from your driving route?" :
                answer = "I am currently {:.3f}m away laterally.".format(lateral_dis_round)

            elif chosen_question == "What should be your next steering action?":
                answer = "My next steering action should be a slight adjustment to {}.".format(action[1])
            elif chosen_question == "What should be your next acceleration command?":
                answer = "My next acceleration command should be {}.".format(action[0])
            elif chosen_question == "Is there any moving object around you?" and vehicles_num == 0:
                answer = "There is no moving object around me."
            elif chosen_question == "Is there any moving object around you?" and vehicles_num == 1:
                answer = "There is a moving object around me."
            elif chosen_question == "Is there any moving object around you?" and vehicles_num > 1:
                answer = "There are {} moving objects around me.".format(vehicles_num)

            elif chosen_question == "Describe the position of the car relative to your heading." and vehicles_num == 0 :
                answer = "There are no cars here."
            elif chosen_question == "Describe the position of the car relative to your heading." and vehicles_num == 1:
                answer = "There is one car here. The car is positioned at an angle of {}° relative to my heading.".format(multi_yaw)
            elif chosen_question == "Describe the position of the car relative to your heading." and vehicles_num > 1:
                answer = "There are {} cars here. Those cars are respectively positioned at angle of {}° relative to my heading.".format(vehicles_num, multi_yaw)

            elif chosen_question == "What is your current lateral position relative to your route?":
                answer = "My current lateral position relative to my route is {:.3f}m.".format(lateral_dis_round)
            elif chosen_question == "What would be a safe driving action given the detected car's details?":
                answer = "A safe driving action would be action=[{}, {}] considering the car's details.".format(action[0], action[1])

            elif chosen_question == "What is the speed of the detected car?" and vehicles_num == 0 :
                answer = "Sorry, there should be no cars here. Please take another careful look."

            elif chosen_question == "What is the speed of the detected car?" and vehicles_num == 1 :
                answer = "The speed of the detected car is {} m/s.".format(speed_round)
            elif chosen_question == "What is the speed of the detected car?" and vehicles_num > 1:
                answer = "The speed of the detected car are respectively {} m/s"

            elif chosen_question == "How far is the detected car from you?" and vehicles_num == 0:
                answer = "Sorry, there should be no cars here. Please take another careful look."
            elif chosen_question == "How far is the detected car from you?" and vehicles_num == 1:
                answer = "The detected car is {} meters away from me.".format(multi_dis)
            elif chosen_question == "How far is the detected car from you?" and vehicles_num > 1:
                answer = "The detected cars are respectively {} meters away from me.".format(multi_dis)


            elif chosen_question == "What angle should you adjust your steering to avoid collision?":
                answer = "I should adjust my steering to {}.".format(action[1])
            elif chosen_question == "Why is it important to note the angle of the detected car?":
                answer = "It is important to note the angle of the detected car to determine the correct steering adjustment for safe driving."
            elif chosen_question == "Is the detected car in motion?":
                in_motion = "yes" if vehicles_num > 0 else "no"
                answer = "The detected car is in motion: {}.".format(in_motion)
            elif chosen_question == "What should you be cautious of given the car’s position?":
                answer = "I should be cautious of maintaining a safe distance and appropriate angle to avoid potential collisions given the car’s position."


            elif chosen_question == "What action should be taken to maintain alignment on your driving route?":
                answer = "To maintain alignment the action should be action=[{}, {}].".format(action[0], action[1])
            elif chosen_question == "What considerations are necessary for the detected car's speed?" and vehicles_num == 0:
                answer = "Although there are no vehicles currently present, I should control my speed appropriately in areas where vehicles are likely to appear for safer driving."
            elif chosen_question == "What considerations are necessary for the detected car's speed?" and vehicles_num == 1:
                answer = "Since the detected car’s speed is {} m/s I should consider it {} and plan my action accordingly.".format(speed_round, "stationary" if len(speed_round) == 0  else "moving")
            elif chosen_question == "What considerations are necessary for the detected car's speed?" and vehicles_num > 1:
                answer = "Since the detected car’s speed are respectively {} m/s I should consider their {} and plan my action accordingly.".format(
                    speed_round, "stationary" if len(speed_round) == 0 else "moving")


            elif chosen_question == "What’s the importance of your current lateral position in planning the next action?":
                answer = "My current lateral position of {:.3f}m helps ensure that my planned action maintains the vehicle’s alignment on the route.".format(lateral_dis_round)
            elif chosen_question == "What factors are influencing your next driving decision?":
                answer = "Factors influencing my next driving decision include the car's distance, angle, speed, and my current lateral position."

            elif chosen_question == "Is there any obstacle directly ahead?" and vehicles_num == 0:
                answer = "No, there is no obstacle directly ahead. The detected car(s) is/are at an angle of {}°.".format(multi_yaw)
            elif chosen_question == "Is there any obstacle directly ahead?" and vehicles_num == 1:
                answer = "The detected car is at an angle of {}°.".format(multi_yaw)
            elif chosen_question == "Is there any obstacle directly ahead?" and vehicles_num > 1:
                answer = "The detected cars are respectively at angle of {}°.".format(multi_yaw)

            elif chosen_question == "How should you interpret the car’s angle for your steering decision?" and vehicles_num == 0:
                answer = "Sorry, there shouldn’t be any cars here. Please revise your question"
            elif chosen_question == "How should you interpret the car’s angle for your steering decision?" and vehicles_num >= 1:
                answer = "The car’s angle of {}° suggests a need for slight steering adjustment to avoid potential collision while maintaining my route.".format(multi_yaw)
            elif chosen_question == "What immediate adjustments are necessary for safe driving?":
                answer = "Immediate adjustments include a slight steering adjustment and maintaining acceleration as indicated by action=[{}, {}].".format(
                    action[0], action[1])

            elif chosen_question == "How does the detected car's speed impact your driving action?" and vehicles_num == 0 :
                answer = "Since there are no cars here, I don’t need to change."
            elif chosen_question == "How does the detected car's speed impact your driving action?"  and vehicles_num > 0:
                answer = "Since the detected car's speed is moving and I need to plan my action to safely navigate around it without sudden changes."

            elif chosen_question == "What should be your focus given the detected car’s proximity and angle?":
                answer = "My focus should be on adjusting my steering and speed appropriately to navigate safely around the car(s)."
            elif chosen_question == "What safe action is suggested based on the current scenario?":
                answer = "Based on the current scenario, the suggested safe action is action=[{}, {}].".format(action[0], action[1])
            elif chosen_question == "What should you avoid in this situation to prevent collision?":
                answer = "I should avoid sudden or large steering adjustments and ensure controlled acceleration to prevent collision."
            elif chosen_question == "Is there a need for a speed adjustment?":
                answer = "Yes, a controlled acceleration to {} is suggested.".format(action[0])
            elif chosen_question == "How will your steering angle change based on the detected car’s angle?":
                answer = "My steering angle will change slightly to {} to align safely given the detected car's angle.".format(action[1])
            elif chosen_question == "What should you consider for maintaining a safe path?":
                answer = "I should consider the car's position, my current lateral alignment, and necessary adjustments in steering and acceleration."
            elif chosen_question == "How would you describe the current traffic conditions?":
                answer = "The traffic condition involves {} stationary car(s) located at a specific distance and angle from my vehicle, requiring careful planning to proceed.".format(
                    vehicles_num)
            elif chosen_question == "What immediate action is necessary given your current lateral position?":
                answer = "Given my current lateral position, the immediate action is to maintain slight adjustments in steering and controlled acceleration."
            elif chosen_question == "What factors need to be monitored to ensure safe navigation?":
                answer = "I need to monitor the distance to the car(s), its angle relative to my heading, and my current lateral alignment."
            elif chosen_question == "Is the detected car influencing your path directly?":
                answer = "No, the car(s) is/are not directly in my path but require careful navigation due to its/their proximity and angle."

            elif chosen_question == "What is the priority in adjusting your speed and direction?":
                answer = "The priority is to adjust my speed and direction to action=[{}, {}] to ensure safe driving.".format(action[0], action[1])
            elif chosen_question == "":
                answer = ""

            modified_answer = re.sub(r"action=\[[\d\.]+,\s*[\d\.]+\]",  "action=[action1, action2]", answer)   
            datasets.append({"input": original_input, "Question": chosen_question, "Answer": answer, "lidar_noground": lidar_ng, "reward": reward, "sum_rewards": sum_rewards, "action": action})

        pickle.dump(datasets, f)
        print('save successfully')

