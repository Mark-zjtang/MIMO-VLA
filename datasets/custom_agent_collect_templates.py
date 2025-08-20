# This is an example for adding sensor inputs templates to Custom Agent(eg. RL or IL agent)
# Below sensor_input_datasets were collected by EGADS agent, for more detailes, please refer to https://github.com/Mark-zjtang/EGADS

qa_dataset = []
# Initialise parallelised test environments
count_frame = 0
test_envs = customer_env # your selected env
for episode in range(0, 200):
  # Set models to eval mode
  transition_model.eval()
  observation_model.eval()
  reward_model.eval()
  encoder.eval()
  actor_model.eval()
  value_model.eval()
  bc_model.eval()
  with torch.no_grad():
    observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
    belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size,device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
    pbar = tqdm(range(args.max_episode_length // 4))
    test_episode_length = 0
    test_collision_distance = 0
    test_collision_times = 0
    vehicle_work_distance = 0
    for t in pbar:
      belief, posterior_state, action, next_observation, reward, done, collision_time, vehicle_collision_distance, observation_vehicle_state = update_belief_and_act(args, test_envs,planner,transition_model,encoder, belief,posterior_state,action.to( device=args.device), observation)
      total_rewards += reward
      contex = "I am now an expert in car driving and my goal is always to keep the vehicle safe. I need to pay attention to and understand the numerical changes on the following trajectories, because this is the biggest difference when the trajectory information changes. According to the following numerical changes of the trajectory information and my driving experience, I need to predict respectively the throttle value, brake value and steering wheel value of the current moment to ensure safely driving and higher reward value in the next time, and the time interval is 0.1 seconds"
      new_instruction = contex
      new_instruction_different_traj = "The current frame and the previous frame do not belong to the same turn, and there is no strong correlation between the two. "
      if done:
        new_instruction = contex + new_instruction_different_traj

      if action[0][0] > 0 :
        throttle = action[0][0]
        brake = 0
      else:
        throttle = 0
        brake = action[0][0]
      new_output = "In order to drive safely, the current moment, my throttle value, brake value, steering_wheel value commands are respectively {:.3f} m/s^2, {:.3f} /s^2, {:.3f} rad \n Here are my actions: \n aciton=[{:.3f},{:.3f}] ".format(throttle, brake, action[0][1],throttle+brake, action[0][1])
      lateral_dis, delta_yaw, speed, vehicles_info = observation_vehicle_state
      vehicles_num = len(vehicles_info)
      multi_dis, multi_yaw, multi_speed =" ", " ", " "
      if vehicles_num >= 1 :
        for i in range(vehicles_num):
          multi_dis +=  str(vehicles_info[i][0]) + " "
          multi_yaw +=  str(vehicles_info[i][1]) + " "
          multi_speed += str(vehicles_info[i][2]) + " "

        new_input = " I can see that there are {} cars, their speed, their straight-line distance from me, and their Angle in the direction my heading are {}m/s, {}m, {}°." \
                              " I am now {:.3f}m laterally away from my driving route.  My yaw Angle throughout the current driving environment is {:.3f}° and my speed is {:.3f}m/s. I got a reward value of {:.3f} after I made a decision command in the last frame. ".format( vehicles_num,multi_speed,multi_dis,multi_yaw, lateral_dis, delta_yaw, speed, reward)
      else:
        new_input = "I see no car here. I am now {:.3f}m laterally away from my driving route.  My yaw Angle throughout the current driving environment is {:.3f}° and my speed is {:.3f}m/s. I got a reward value of {:.3f} after I made a decision command in the last frame".format( lateral_dis, delta_yaw, speed, reward)
      qa_dataset.append({"instruction": new_instruction, "input": new_input, "output": new_output})
      count_frame += 1

      if args.render:
        test_envs.render()
      if done:
        pbar.close()
        break

  with open("sensor_input_dataset{}.json".format(args.dataset), "w", encoding="utf-8") as json_file:
  json.dump(qa_dataset, json_file, ensure_ascii=False, indent=4)
  env.close()

