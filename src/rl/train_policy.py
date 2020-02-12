from dm_control import mujoco
# Load a model from an MJCF XML string.
xml_string = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1.5"/>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="box" pos="0 0 .3">
      <joint name="up_down" type="slide" axis="0 0 1"/>
      <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

physics = mujoco.Physics.from_xml_string(xml_string)
# Render the default camera view as a numpy array of pixels.
pixels = physics.render()

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from datetime import datetime
from PIL import Image
import sys
from collections import namedtuple
sys.path.insert(0, '/u/pgoyal/Research/metaworld')
sys.path.insert(0, '../supervised')
# sys.path.insert(0, '/u/pgoyal/Research/metaworld/supervised')
from metaworld.envs.mujoco.sawyer_xyz.sawyer_random import SawyerRandomEnv
from model import Predict
import pickle
from ppo import Memory, ActorCritic, PPO

def recreate_env(infile):
    objects = [ 
                'button_top', 
                'button_side', 
                'coffee_button', 
                'handle_press_top',
                'handle_press_side',
                'door_lock',
                'door_unlock',
                'dial_turn',
                'faucet_open',
                'faucet_close',
                'window_open',
                'window_close',
                'peg_unplug',
              ]
    positions = []
    obj_ids = []
    with open(infile) as f:
        for line in f.readlines():
            line = line.replace('(', '').replace(',', '').replace(')', '')
            parts = line.split()
            x = eval(parts[0])
            y = eval(parts[1])
            obj = eval(parts[2])
            print(x, y, obj)
            positions.append((x, y))
            obj_ids.append(obj)

    return objects, positions, obj_ids

def encode_description(vocab, descr):
    result = []
    for w in descr.split():
            try:
                    t = vocab.index(w)
            except ValueError:
                    t = vocab.index('<unk>')
            result.append(t)
    return torch.Tensor(result)

def load_descriptions(vocab, mode):
    descriptions = pickle.load(open('../../data/{}_descr.pkl'.format(mode), 'rb'))
    result = {}
    for i in descriptions.keys():
            descr_list = descriptions[i]
            result[i] = [(d, encode_description(vocab, d)) for d in descr_list]
    return result

def main(args):
    ############## Hyperparameters ##############
    render = False
    solved_reward = 1e10         # stop training if avg_reward > solved_reward
    log_interval = 1000           # print avg reward in the interval
    save_interval = 500
    max_episodes = 10000        # max training episodes
    max_timesteps = args.max_timesteps
    max_total_timesteps = args.max_total_timesteps
    if max_total_timesteps == 0:
        max_total_timesteps = np.inf
    
    update_timestep = args.update_timestep # update policy every n timesteps
    action_std = args.action_std            # constant std for action distribution (Multivariate Normal)
    K_epochs = args.K_epochs               # update policy for K epochs
    eps_clip = args.eps_clip              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    # random_seed = 17
    random_seed = None
    #############################################

    # creating environment
    objects, positions, obj_ids = recreate_env(
        '../../data/envs/obj{}-env{}.txt'.format(args.main_obj, args.env_id))
    state_dim = 6
    action_dim = 4

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(args, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    directory = "./policies-grid-all/"
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    success_list = []
    
    env = SawyerRandomEnv(
        objects=objects, 
        positions=positions, 
        obj_ids=obj_ids, 
        state_rep='feature', 
        sparse_reward=args.sparse,
        max_timesteps = max_timesteps)
    if args.langreward:
        lang_network = Predict('../supervised/model.pt', lr=0, n_updates=0)
    vocab = pickle.load(open('../../data/vocab_train.pkl', 'rb'))
    valid_descr = load_descriptions(vocab, 'test')
    t = args.trial
    descr = valid_descr[args.main_obj][t][1]
    if args.drop >= 0:
        descr = torch.cat([descr[:args.drop], descr[args.drop+1:]])
    print(descr)
    total_steps = 0
    i_episode = 0
    while True:
        i_episode += 1
        traj_r = []
        traj_l = []
        traj_c = []
        lang_rewards = []
        state = env.reset()
        img_left, img_center, img_right, _ = env.get_frame()
        img_left = Image.fromarray(img_left)
        img_left = np.array(img_left.resize((50, 50)))
        img_center = Image.fromarray(img_center)
        img_center = np.array(img_center.resize((50, 50)))
        img_right = Image.fromarray(img_right)
        img_right = np.array(img_right.resize((50, 50)))
        traj_r.append(img_right)
        traj_l.append(img_left)
        traj_c.append(img_center)
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, success = env.step(action)
            img_left, img_center, img_right, _ = env.get_frame()
            img_left = Image.fromarray(img_left)
            img_left = np.array(img_left.resize((50, 50)))
            img_center = Image.fromarray(img_center)
            img_center = np.array(img_center.resize((50, 50)))
            img_right = Image.fromarray(img_right)
            img_right = np.array(img_right.resize((50, 50)))
            if args.langreward:
                traj_r.append(img_right)
                traj_l.append(img_left)
                traj_c.append(img_center)
                if done:
                    prob = 0
                else:
                    prob = torch.tanh(lang_network.predict(
                        traj_r, traj_l, traj_c, descr)).data.cpu().numpy()[0][0]
                lang_rewards.append(prob)

                if len(lang_rewards) > 1:
                    reward += (gamma * lang_rewards[-1] - lang_rewards[-2])
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                total_steps += time_step
                time_step = 0
            running_reward += reward
            if render:
                env.render()


            if (total_steps + time_step) % log_interval == 0:
                current_time = datetime.now().strftime('%H:%M:%S')
                print('[{}] \t Episode {} \t Timesteps: {} \t Success: {}'.format(
                    current_time, i_episode, total_steps + time_step, sum(success_list)))
            if done:
                success_list.append(success)
                break
        
            if total_steps + time_step >= max_total_timesteps:
                break
        
        if total_steps + time_step >= max_total_timesteps:
            break
        
        if args.save_path:
            if sum(success_list[-5:]) == 5:
                torch.save(ppo.policy.state_dict(), args.save_path)
                break

        
        if args.model_file and i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), args.model_file)
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Train PPO policy')
    parser.add_argument('--random-init', type=int, help='Environment seed')
    parser.add_argument('--sparse', action='store_true', help='use sparse rewards')
    parser.add_argument('--langreward', action='store_true', help='use language-based rewards')
    parser.add_argument('--model-file', help='')
    parser.add_argument('--save-path', help='')
    parser.add_argument('--main-obj', type=int, help='Index of main object; 0-12')
    parser.add_argument('--env-id', type=int, help='Index of environment; 0-99')
    parser.add_argument('--trial', type=int, help='')
    parser.add_argument('--n-channels', type=int, default=64)
    parser.add_argument('--img-enc-size', type=int, default=128)
    parser.add_argument('--lang-enc-size', type=int, default=128)
    parser.add_argument('--max-timesteps', type=int, default=500)
    parser.add_argument('--max-total-timesteps', type=int, default=500000)
    parser.add_argument('--update-timestep', type=int, default=2000)
    parser.add_argument('--action-std', type=float, default=0.5)
    parser.add_argument('--K-epochs', type=int, default=10)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--noise', type=float, default=0.)
    parser.add_argument('--drop', type=int, default=-1)
 
    args = parser.parse_args()
    main(args)
    
