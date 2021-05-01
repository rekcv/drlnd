### p1_navigation (5/1/2021)


## Project Details:

This project presents an implementation of DQN to solve a simplified version of the Banana Collector Environment from Unity's ML-Agents.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

Our solution has been able to solve the environment in 423 episodes, with an average score of 13.03 over the last 100 episodes of the training.

## Getting started

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

After that, please download the version of the Unity Environment that corresponds to your OS.

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
[MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
[Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
[Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Clone [my repository](https://github.com/rekcv/drlnd) and extract the environment in the p1_navigation folder (my repository already includes a copy of the Win64 version).

**The solution can be run via p1_navigation\Report.ipynb**

