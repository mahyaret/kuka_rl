import gym
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p


class ContinuousDownwardBiasPolicy(object):
  """Policy which takes continuous actions, and is biased to move down.
  """

  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.
    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(5,))
    

  def sample_action(self, obs, explore_prob):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da, close = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da, 0]




env = KukaDiverseObjectEnv(renders=True, isDiscrete=False, removeHeightHack=True)
env.cid =  p.connect(p.DIRECT)
policy = ContinuousDownwardBiasPolicy()

while True:
    obs, done = env.reset(), False
    print("===================================")
    print("obs")
    print(obs)
    episode_rew = 0
    while not done:
      env.render(mode='human')
      act = policy.sample_action(obs, .1)
      #act = [0,0,-1,0,100]
      print("Action")
      print(act)
      obs, rew, done, _ = env.step(act)
      episode_rew += rew
      print("Ac rew:", episode_rew)
    print("Episode reward", episode_rew)
