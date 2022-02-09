This repository is being used to upload the different projects I have worked on for Udacity's Deep Reinforcement Learning Nanodegree.

2/9/2022
--------
p2_continuous-control: Contains an implementation of DDPG to solve the continuous control problem (Unity Reacher Environment - Multiagent)

# p2_continuous-control (2/9/2022)
## Project Details:

This project presents an implementation of DDPG to solve a simplified version of the Multiagent Reacher Environment from Unity's ML-Agents.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

Our solution has been able to solve the environment in 562 episodes, with an average score of 13.06 over the last 100 episodes of the training.

## Getting started

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

After that, please download the version of the Unity Environment that corresponds to your OS.

Version 2: Twenty (20) Agents
[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
[Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
[Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
[Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Clone [my repository](https://github.com/rekcv/drlnd) and extract the environment in the p1_navigation folder (my repository already includes a copy of the Win64 version).

**The solution can be run via p2_continuous-control\Report.ipynb**


5/1/2021
--------
p1_navigation: Contains an implementation of DQN to solve the navigation problem (Unity Banana Collector Environment).

# p1_navigation (5/1/2021)
## Project Details:

This project presents an implementation of DQN to solve a simplified version of the Banana Collector Environment from Unity's ML-Agents.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

Our solution has been able to solve the environment in 562 episodes, with an average score of 13.06 over the last 100 episodes of the training.

## Getting started

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

After that, please download the version of the Unity Environment that corresponds to your OS.

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
[MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
[Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
[Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Clone [my repository](https://github.com/rekcv/drlnd) and extract the environment in the p1_navigation folder (my repository already includes a copy of the Win64 version).

**The solution can be run via p1_navigation\Report.ipynb**

