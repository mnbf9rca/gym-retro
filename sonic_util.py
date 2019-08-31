"""
Environments and wrappers for Sonic training.
https://github.com/openai/retro-baselines/tree/master/agents
adapted by mnbf9rca
"""
from math import log1p
import gym
import numpy as np
import retro

# from baselines.common.atari_wrappers import WarpFrame, FrameStack


def make_env(game_name, game_level, save_game=False, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    print(f"Creating game '{game_name}' on level '{game_level}' recording to '{save_game}'")
    env = retro.make(
        game=game_name,  # Game
        state=game_level,  # Level / State
        record=save_game)  # Record the Run
    env = SonicDiscretizer(env)
    if scale_rew:
        print("scaling rewards")
        env = RewardScaler(env)
    # env = WarpFrame(env)
    # if stack:
    #     env = FrameStack(env, 4)
    return env


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        '''
        originals from sonic

        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        '''

        '''
        CHASE HQ
        lookimng at https://gamefaqs.gamespot.com/genesis/586101-chase-hq-ii/faqs/16869
        keys are:
        Buttons:

        A button: Brake
        B button: Accelerator
        C button: Turbo button- to activate a speed enhancement

        Dpad: 

        Left: To move car to the left
        Right: To move car to the right
        UP and Down: Nothing changes on movement of the car
        '''        
        # buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # actions = [['A'], ['B'], ['C'], ['B', 'LEFT'], ['B', 'RIGHT']]

        '''
        AIRSTRIKER
        '''
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['X']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        print(f"Initialized {len(self._actions)} discrete actions.")

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        '''returns (log1p(reward), reward)
            rewards 1 for each step which happens to reward duration of game
        '''

        return (float(log1p(max(reward, 1))), reward)


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
