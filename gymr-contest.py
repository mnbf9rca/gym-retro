
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

num_episodes = 100
max_steps = 5000000  # per episode
BATCH_SIZE = 64
REPLAY_CAPACITY = 5000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000  # how many steps does it take before EPS is zero? - across episodes...
TARGET_UPDATE = 10
IMAGE_RESIZED_TO = 160  # squaere
GAME_NAME = 'ChaseHQII-Genesis'
LEVEL = 'Sports.DefaultSettings.Level1'
store_model = True
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
    def __init__(self, d, h, w, number_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(d, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, number_actions)


    def forward(self, x):
        '''
        Called with either one element to determine next action, or a batch during optimization.
        Returns tensor([[left0exp,right0exp]...]).
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# use replay to handle image transitions

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
        '''returns a random sample of batch_size transitions'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


'''
removing greyscale...
resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize((IMAGE_RESIZED_TO, IMAGE_RESIZED_TO)),
                    T.ToTensor()])
'''

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize((IMAGE_RESIZED_TO, IMAGE_RESIZED_TO), interpolation=Image.CUBIC),
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
    fetches a screen from the game AND resizes to IMAGE_RESIZED_TO X IMAGE_RESIZED_TO
    '''
    # return an nparray with order w,h,c
    screen = env.render(mode='rgb_array')
    # plot_image(screen)

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # optionally, reorder to h,w,c (i.e. rotate)
    # screen = screen.transpose((1,0,2))

    # plot_image(screen.cpu().numpy().squeeze())

    return resize(screen).unsqueeze(0).to(device)


def select_action(state, bias_list=None):
    '''
    selects an action for a given state, randomly choosing an action some proportion of the time
    adapted from udacity
    '''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    selected_action = None
    if sample > eps_threshold:
        with torch.no_grad():
            # print("using net")
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = policy_net(state).max(1)[1].view(1, 1)
    else:
        # print("random action")
        # check if there's a bias towards any specific action
        # if bias_list:
        selected_action = torch.tensor([np.random.choice(
                                        env.action_space.n,
                                        1,
                                        p=bias_list)],
                                       device=device,
                                       dtype=torch.long)
        # else:
        # selected_action = torch.tensor(
        #    [[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
    # print(f"selected action {selected_action}")
    return selected_action


def optimize_model():
    '''
    performs a backward pass
    '''
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)  # will throw error on torch > 1.2
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

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return

# wrap the main training loop in a definition


frames = []


def dqn_training(num_episodes, max_steps=500, display_action=False):
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
        # last_screen = get_screen()
        # current_screen = get_screen()
        # state = current_screen - last_screen
        state = get_screen().to(device)
        total_reward = 0
        episode_start_time = datetime.now()
        for t in count():
            # Select and perform an action
            # action = select_action(state, SELECT_ACTION_BIAS_LIST)
            action = select_action(state, SELECT_ACTION_BIAS_LIST)
            # "action", action)
            if display_action:
                print("action: ", action.squeeze())
            _, reward, done, _ = env.step(action)
            total_reward += reward * 10

            reward = torch.tensor([reward], device=device)

            # Observe new state
            #last_screen = current_screen
            #current_screen = get_screen().to(device)
            if not done:
                next_state = get_screen().to(device)
                next_state = next_state.detach()
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state.detach(), action.detach(), next_state, reward.detach())

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done or t > max_steps:
                episode_end_time = datetime.now()
                episode_time = (episode_end_time - episode_start_time).total_seconds()
                # floyd metrics
                print(f'{{"metric": "score", "value": {total_reward}, "epoch": {i_episode+1}}}')
                print(f'{{"metric": "total steps", "value": {steps_done}, "epoch": {i_episode+1}}}')
                print(f'{{"metric": "steps this episode", "value": {t}, "epoch": {i_episode+1}}}')
                print(f'{{"metric": "episode duration", "value": {episode_time}, "epoch": {i_episode+1}}}')
                # paperspace
                # {"chart": "<identifier>", "y": <value>, "x": <value>}
                print(f'{{"chart": "score", "y": {total_reward}, "x": {i_episode+1}}}')
                print(f'{{"chart": "steps_this_episode", "y": {t}, "x": {i_episode+1}}}')
                print(f'{{"chart": "episode_duration", "y": {episode_time}, "x": {i_episode+1}}}')
                break

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
    store_model = False
    BATCH_SIZE = 8
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"setting device to '{device}'")

# close current environment if there is one (e.g. on failure to complete last time)
try:
    env.close()
except NameError:
    pass

# create the environment
# Loading the level
env = make_env(GAME_NAME, LEVEL, True)


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, screen_depth, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_depth, screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_depth, screen_height, screen_width, n_actions).to(device)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(REPLAY_CAPACITY)


steps_done = 0

# charts for paperspace
print('{"chart": "score", "axis": "epoch"}')
print('{"chart": "steps_this_episode", "axis": "epoch"}')
print('{"chart": "episode_duration", "axis": "epoch"}')


dqn_training(num_episodes,
             max_steps=max_steps,
             display_action=display_action)


# save models
if store_model:
    print('saving')
    date_time = datetime.now().strftime("%Y%d%Y-%H%M%S")
    torch.save(target_net.state_dict(), f'models/target_net-{GAME_NAME}-{date_time}.pt')
    torch.save(policy_net.state_dict(), f'models/policy_net-{GAME_NAME}-{date_time}.pt')
    print('saved')
