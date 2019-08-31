
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
from torch.autograd import Variable  # FQDN
from srsly import json_dumps

num_episodes = 1500
max_steps = 5000000  # per episode
BATCH_SIZE = 256
REPLAY_CAPACITY = 5000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 30000  # how many steps does it take before EPS is zero? - across episodes...
TARGET_UPDATE = 10
IMAGE_RESIZED_TO = 80  # squaere
GAME_NAME = 'Airstriker-Genesis' # 'ChaseHQII-Genesis'
LEVEL = 'Level1'# 'Sports.DefaultSettings.Level1'
store_model = True
ROM_PATH = './roms/'  # where to find ROMs
RECORD_DIR = './bk2/'  # where to save output BK2 files
MODEL_DIR = "./models/"  # where to store final model
STATE_DIR = "./gamestates/"  # where to store game state files
report_mean_score_over_n = 50
# or None - bias random selection towards this value
SELECT_ACTION_BIAS_LIST = [0.25, 0.25, 0.15, 0.15, 0.2] # [0.125, 0.25, 0.125, 0.25, 0.25] (remove bias for airstriker)
display_action = False

# Before running the installation steps, we have to check the python version because `gym-retro` doesn't support Python 2.
if ((sys.version_info[0] == 3) & (sys.version_info[1] < 6)) or (sys.version_info[0] < 3):
    raise Exception("Gym Retro requires Python >= 3.6")
else:
    print('Your python version is OK.')
    print(sys.version_info)


# check if running in paperspace
if os.path.isdir("/storage") & os.path.isdir("/artifacts"):
    # /storage and /artifacts both exist
    # these are special directories on paperspace
    print("discovered /storage and /artifacts")
    ROM_PATH = '/storage/roms'
    RECORD_DIR = "/artifacts/bk2"
    MODEL_DIR = "/artifacts/models"
    STATE_DIR = "/artifacts/gamestates"

print(f'Saving to: ROM_PATH="{ROM_PATH}", '
      'RECORD_DIR="{RECORD_DIR}", '
      'MODEL_DIR="{MODEL_DIR}", '
      'STATE_DIR="{STATE_DIR}"')

print("creating RECORD_DIR, MODEL_DIR, STATE_DIR if they dont exist and checking they're writeable")
def check_path_create_if_not(path):
    filename = os.path.join(path, 'dummy_file')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write('this is a test file')
        f.close()
    os.remove(filename)
    print(f'... "{os.path.dirname(filename)}" is ok')
    


check_path_create_if_not(RECORD_DIR)
check_path_create_if_not(MODEL_DIR)
check_path_create_if_not(STATE_DIR)

# Load ROMs
install_games_from_rom_dir(ROM_PATH)


class FDQN(nn.Module):
    '''
    3 CNN + 2 FC layers
    from https://github.com/Shmuma/rl/blob/ptan/ptan/samples/dqn_expreplay_doom.py
    ideal input_shape=(1, 80, 80) (from source)
    '''

    def __init__(self, depth, height, width, n_actions):
        super(FDQN, self).__init__()
        self.conv1 = nn.Conv2d(depth, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 2)

        n_size = self._get_conv_output((depth, height, width))

        self.fc1 = nn.Linear(n_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        print("Conv out shape: %s" % str(output_feat.size()))
        n_size = output_feat.data.view(1, -1).size(1)
        print("Conv out size: %s" % str(n_size))
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(x.size(0), -1)
        # print("forward x", x)
        return x


class DQN(nn.Module):
    '''
    3 or 5 layer CNN, no FC layers
    '''

    def __init__(self, depth, height, width, number_actions):
        super(DQN, self).__init__()
        self.input_width = width
        self.input_height = height
        self.input_depth = depth
        self.conv1 = nn.Conv2d(self.input_depth, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        if self.input_width >= 128 and self.input_height >= 128:
            self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn4 = nn.BatchNorm2d(32)
            self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn5 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_height)))
        if self.input_width >= 128 and self.input_height >= 128:
            convw = conv2d_size_out(conv2d_size_out(convw))
            convh = conv2d_size_out(conv2d_size_out(convh))

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
        if self.input_width >= 128 and self.input_height >= 128:
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
        x = self.head(x.view(x.size(0), -1))
        # print("forward x", x)
        return x


class scoreAverage():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, score):
        """Saves a score."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = score
        self.position = (self.position + 1) % self.capacity

    def mean(self):
        if len(self.memory) == 0:  # pylint: disable=C1801
            return None
        return float(sum(self.memory))/float(len(self.memory))

# use replay to handle image transitions


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():

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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
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
    score_history = scoreAverage(report_mean_score_over_n)
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        # state = get_screen().to(device)
        total_reward = 0
        episode_start_time = datetime.now()
        statememory = []
        for t in count():
            # initialise state memory

            # Select and perform an action
            # action = select_action(state, SELECT_ACTION_BIAS_LIST)
            action = select_action(state, SELECT_ACTION_BIAS_LIST)
            # "action", action)
            if display_action:
                print("action: ", action.squeeze())
            _, reward, done, info = env.step(action)

            # ('action', 'reward', 'info', 'done')
            this_state = {"action": action.data[0].item(
            ), "reward": reward[1], "scaled_reward": reward[0], "info": info, "done": done}

            statememory.append(this_state)
            total_reward += reward[1]

            reward = torch.tensor([reward[0]], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen().to(device)
            if not done:
                next_state = current_screen
                next_state = next_state
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done or t > max_steps:
                episode_end_time = datetime.now()
                episode_time = (episode_end_time - episode_start_time).total_seconds()
                score_history.push(total_reward)
                # floyd metrics
                print(f'{{"metric": "score", "value": {total_reward}, "epoch": {i_episode+1}}}')
                print(
                    f'{{"metric": "rolling mean score", "value": {score_history.mean()}, "epoch": {i_episode+1}}}')
                print(f'{{"metric": "steps this episode", "value": {t}, "epoch": {i_episode+1}}}')
                print(f'{{"metric": "episode duration", "value": {episode_time}, "epoch": {i_episode+1}}}')
                print(
                    f'{{"metric": "steps per second", "value": {float(t) / float(episode_time)}, "epoch": {i_episode+1}}}')

                # paperspace
                # {"chart": "<identifier>", "y": <value>, "x": <value>}
                print(f'{{"chart": "score", "y": {total_reward}, "x": {i_episode+1}}}')
                print(f'{{"chart": "rolling_mean_score", "y": {score_history.mean()}, "x": {i_episode+1}}}')
                print(f'{{"chart": "steps_this_episode", "y": {t}, "x": {i_episode+1}}}')
                print(f'{{"chart": "episode_duration", "y": {episode_time}, "x": {i_episode+1}}}')
                print(
                    f'{{"chart": "steps_per_second", "y": {float(t) / float(episode_time)}, "x": {i_episode+1}}}')
                filename = os.path.join(STATE_DIR,
                                        f'gamedata-{GAME_NAME}-{LEVEL}-{(i_episode+1):06}.json')
                print(f"Writing game history to '{filename}'")
                with open(filename, "w") as f:
                    f.write(json_dumps(statememory))
                    f.close()

                break

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Completed training')

    # env.render(close=True)
    env.close()


# set to GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    # limit episides, steps on CPU i.e. when testing locally
    num_episodes = 3
    max_steps = 500
    store_model = False
    RECORD_DIR = False
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
env = make_env(GAME_NAME, LEVEL, save_game=RECORD_DIR)


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, screen_depth, screen_height, screen_width = init_screen.shape
print(f"discovered input image ({screen_depth},{screen_height},{screen_width})")

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = FDQN(screen_depth, screen_height, screen_width, n_actions).to(device)
target_net = FDQN(screen_depth, screen_height, screen_width, n_actions).to(device)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(REPLAY_CAPACITY)


steps_done = 0

# charts for paperspace
print('{"chart": "score", "axis": "epoch"}')
print('{"chart": "rolling_mean_score", "axis": "epoch"}')
print('{"chart": "steps_this_episode", "axis": "epoch"}')
print('{"chart": "episode_duration", "axis": "epoch"}')
print('{"chart": "steps_per_second", "axis": "epoch"}')


dqn_training(num_episodes,
             max_steps=max_steps,
             display_action=display_action)


# save models
if store_model:
    print('saving')
    date_time = datetime.now().strftime("%Y%d%Y-%H%M%S")
    torch.save(target_net.state_dict(),
               os.path.join(MODEL_DIR, f'target_net-{GAME_NAME}-{date_time}.pt'))
    torch.save(policy_net.state_dict(),
               os.path.join(MODEL_DIR, f'policy_net-{GAME_NAME}-{date_time}.pt'))
    print('saved')
