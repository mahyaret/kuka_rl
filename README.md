# Robotic Grasping
Machine learning implementations using OpenAI Gym and PyBullet for robotic grasping.

<img src="img/kuka.gif" height="200">

## Deep Q Learning (DQN) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahyaret/kuka_rl/blob/master/kuka_rl.ipynb)
* Discrete control problem is considered.
* Based on [PyTorch tutorial example](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
* [kuka_rl](https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl.ipynb)


## Proximal Policy Optimization (PPO) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahyaret/kuka_rl/blob/master/kuka_rl_2.ipynb)
* Continuous control problem is considered.
* [kuka_rl_2](https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl_2.ipynb)
#### A2C Parallel Implementation
* Twice the number of available CPU threads is considered for the KUKA environment to run in parallel.
* [kuka_rl_2_parallel](https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl_2_parallel.ipynb)


### Dependencies
```
pip install pybullet
pip install gym
```
