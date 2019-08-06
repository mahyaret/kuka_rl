import gym
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p
from collections import defaultdict


class FirstVisitMonteCarlo(object):
  """Policy which takes continuous actions, and is biased to move down.
  """

  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.
    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self.env = KukaDiverseObjectEnv(renders=True, isDiscrete=False, removeHeightHack=True, maxSteps=20)
    self.env.cid = p.connect(p.DIRECT)
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(5, 1))

  def estimate_Q(self, num_episodes, gamma):
    return_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(self._action_space.shape)))
    q_function = defaultdict(lambda: defaultdict(lambda: np.zeros(self._action_space.shape)))
    N = defaultdict(lambda: defaultdict(lambda: np.zeros(self._action_space.shape)))
    for i in range(num_episodes):
      state = self.env.reset()
      G = 0
      while True:
        self.env.render(mode='human')
        action = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
          action[2] = -1
        action_r = np.round(action)
        state, reward, done, info = self.env.step(action_r)
        #print(state.shape, " ", action.shape)
        G = gamma*G + reward
        #print(G)
        state_m = tuple(np.round(state.flat))
        action_m = tuple(action_r.flat)
        return_sum[state_m][action_m] = return_sum[state_m][action_m] + G
        N[state_m][action_m] = N[state_m][action_m] + 1
        if done:
          break
        #print("return: ", return_sum[state_m][action_m], " N: ", N[state_m][action_m])
        q_function[state_m][action_m] = return_sum[state_m][action_m]/N[state_m][action_m]
    return q_function

  

  def generate_episode_from_Q(self, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = self.env.reset()
    while True:
      self.env.render(mode='human')
      state_m = tuple(np.round(state.flat))
      action = np.random.choice(np.arange(nA), p=get_probs(Q[state_m], epsilon, nA)) if state_m in Q else self._action_space.sample()
      if np.random.random() < self._height_hack_prob:
        action[2] = -1
      if state_m in Q:
        print("*******************From Q!*************************") 
      action_r = np.round(action)
      next_state, reward, done, info = self.env.step(action_r)
      episode.append((state_m, action_r, reward))
      state = next_state
      if done:
        break
    return episode

  def update_Q(self, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
      #print(state)
      old_Q = Q[state][tuple(np.round(actions[i].flat))]
      Q[state][tuple(np.round(actions[i].flat))] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q

    
  def mc_control(self, num_episodes, alpha, gamma=0.95, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = self._action_space.shape[0]
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = self.generate_episode_from_Q(Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = self.update_Q(episode, Q, alpha, gamma)
        # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q

  


  def sample_action(self, obs, explore_prob):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da, close = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da, 0]


if __name__ == '__main__':
  kuka = FirstVisitMonteCarlo()
  #action_value_function = kuka.estimate_Q(10, 0.95)
  kuka.mc_control(num_episodes=10000, alpha=0.1)
  
