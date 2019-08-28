
# coding: utf-8

# # Gym Retro
#
# Based on [Floyd Hub](https://www.floydhub.com)'s [Gym Retro](https://blog.openai.com/gym-retro/) "get started" [template](https://github.com/floydhub/gym-retro-template) which was inspired by [this contest](https://blog.openai.com/retro-contest/). You can find more detail in this [page](https://contest.openai.com/details).

# ## Initial Setup

# hyperparameters
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
from PIL import Image
from datetime import datetime
import math
import random
import numpy as np
from itertools import count
import retro
import os
from support import save_frames_as_gif, install_games_from_rom_dir, download_and_unzip_rom_archive_from_url
from sonic_util import make_env


BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000  # how many steps does it take before EPS is zero? - across episodes...
TARGET_UPDATE = 10
GAME_NAME = 'ChaseHQII-Genesis'
LEVEL = 'Sports.DefaultSettings.Level1'
NUMBER_GAME_BUTTONS = 5
num_episodes = 1000
max_steps = 5000000  # per episode
# or None - bias random selection towards this value
SELECT_ACTION_BIAS_LIST = [0.175, 0.3, 0.175, 0.175, 0.175]
display_action = False


# Before running the installation steps, we have to check the python version because `gym-retro` doesn't support Python 2.

if sys.version_info[0] < 3:
    raise Exception("Gym Retro requires Python > 2")
else:
    print('Your python version is OK.')
    print(sys.version_info)


# Load ROMs
DS_PATH = 'roms/'  # edit with your path/to/rom

install_games_from_rom_dir(DS_PATH)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 160, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(160, 80, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(
            nn.Linear(32000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, NUMBER_GAME_BUTTONS))

        # self.head = nn.Linear(12, 12) # 12 possible actions in sonic 2

    def forward(self, x):
        out = x.to(device)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out

    def predict(self, x):
        ''' This function for predicts classes by calculating the softmax '''
        logits = self.forward(x.to(device))
        # logits = F.softmax(logits)
        indices = torch.argmax(logits, dim=1)
        return indices.to(device)


# use replay to handle image transitions

# In[15]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[17]:
resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize((320, 320)),
                    T.ToTensor()])

# notes on output image dimensions/tensor layout
# T.ToPILImage()
#     Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
#     H x W x C to a PIL Image while preserving the value range.

# env.render()
#        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
#          representing RGB values for an x-by-y pixel image, suitable
#          for turning into a video.


def get_screen():
    '''
    fetches a screen from the game, converts to greyscale and resizes to 320x320
    '''
    # return an nparray with order w,h,c
    screen = env.render(mode='rgb_array')
    # plot_image(screen)
    # optionally, reorder to h,w,c (i.e. rotate)
    # screen = screen.transpose((1,0,2))
    screen = resize(screen)
    # plot_image(screen.cpu().numpy().squeeze())

    return screen.unsqueeze(0)


def select_action(state, bias_list=None):
    '''
    selects an action for a given state, randomly choosing an action some proportion of the time
    adapted from udacity
    '''
    global steps_done
    # print("selecting action")
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    selected_action = None
    if sample > eps_threshold:
        with torch.no_grad():
            # print("using net")
            # return policy_net(state).max(1)[1].view(1, 1)
            selected_action = policy_net.predict(state)
    else:
        # print("random action")
        # check if there's a bias towards any specific action
        if bias_list:
            selected_action = torch.tensor(np.random.choice(
                NUMBER_GAME_BUTTONS, 1, p=bias_list), device=device, dtype=torch.long)
        else:
            selected_action = torch.tensor(
                [random.randrange(NUMBER_GAME_BUTTONS)], device=device, dtype=torch.long)
    # print(f"selected action {selected_action}")
    return selected_action


def optimize_model():
    '''
    performs a backward pass
    '''
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    print('{{"metric": "loss", "value": {}}}'.format(loss))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[21]:


# wrap the main training loop in a definition

frames = []


def dqn_training(num_episodes, visualize_plt=False, max_steps=500, report_every=5000, display_action=False):
    """
    num_episodes: int 
        number of episodes
    visualize_plt: bool
        if true, display the cartpole action in the notebook
        if false (default), display the episodes x durations graph
    """
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen().to(device)
        current_screen = get_screen().to(device)
        state = current_screen - last_screen
        iteration_counter = 0
        dt_last_iteration = datetime.now()
        t_last = 0
        total_reward = 0
        for t in count():

            # Select and perform an action
            action = select_action(state, SELECT_ACTION_BIAS_LIST)
            if display_action:
                print("action: ", action.squeeze())
            observation, reward, done, info = env.step(action)
            total_reward += reward * 100

            # frames.append(observation) # collecting observation

            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen().to(device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done or t > max_steps:
                episode_durations.append(t + 1)
                print(f'{{"metric": "score", "value": {total_reward}, "epoch": {i_episode+1}}}')
                observation = env.reset()
                # frames.append(observation)
                # now = datetime.now() # current date and time
                #date_time = now.strftime("%Y%d%Y-%H%M%S")
                #save_frames_as_gif(frames, filename=f'sonic2-{date_time}.gif')
                break
            iteration_counter += 1
            if iteration_counter >= report_every:
                now = datetime.now()  # current date and time
                compute_time = (now - dt_last_iteration).total_seconds()
                t_this = t - t_last
                its_per_sec = float(t_this) / compute_time

                date_time = now.strftime("%Y-%d-%Y %H:%M:%S")
                print(f"t@{date_time}: {t+1} ({its_per_sec} / sec at {compute_time} seconds)")
                #date_time = now.strftime("%Y%d%Y-%H%M%S")
                #save_frames_as_gif(frames, filename=f'sonic2-{date_time}.gif')
                # frames.clear()
                iteration_counter = 0
                t_last = t
                dr_last_iteration = now
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

    # env.render(close=True)
    env.close()


# set to GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    # limit episides, steps on CPU i.e. when testing locally
    num_episodes = 1
    max_steps = 500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"setting device to '{device}'")


policy_net = DQN().to(device)
target_net = DQN().to(device)
# print("Model's state_dict:")
# for param_tensor in policy_net.state_dict():
#     print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10)


steps_done = 0
episode_durations = []


# close current environment if there is one (e.g. on failure to complete last time)
try:
    env.close()
except NameError:
    pass

# create the environment
# Loading the level
env = make_env(GAME_NAME, LEVEL, True)
report_every = max(max_steps/100, 100)

dqn_training(num_episodes, max_steps=max_steps,
             report_every=report_every, display_action=display_action)


# save models
print('saving')
date_time = datetime.now().strftime("%Y%d%Y-%H%M%S")
torch.save(target_net.state_dict(), f'models/target_net-{GAME_NAME}-{date_time}.pt')
torch.save(policy_net.state_dict(), f'models/policy_net-{GAME_NAME}-{date_time}.pt')
print('saved')
