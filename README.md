# nav-env

A OpenAI-gym compatible navigation simulator, which can be integrated into the robot operating system (ROS) with the goal for easy comparison of various approaches including state-of-the-art learning-based approaches and conventional ones.

![1.gif](assets/indoor.gif)
![1.gif](assets/outdoor.gif)

## Install

The package has been tested on Ubuntu 18.04 / Python 3.6 / ROS Melodic.

To install without ROS-supported:  

```bash
virtualenv ~/venv/hrlnav --python=python3.6
source ~/venv/hrlnav/bin/activate

git clone https://github.com/leekwoon/nav_gym.git

cd nav-gym/nav_gym
pip install -e .
```

To install with ROS-supported:  

```bash
virtualenv ~/venv/hrlnav --python=python3.6
source ~/venv/hrlnav/bin/activate

cd ~/YOUR_CATKIN_WORKSPACE/src
git clone https://github.com/leekwoon/nav_gym.git

# install nav_gym_env
cd nav-gym/nav_gym
pip install -e .

cd ~/YOUR_CATKIN_WORKSPACE
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash # or you can write it on ~/.bashrc
```

## Usage

```python
import gym
import nav_gym_env

env = gym.make("NavGym-v0")
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample() # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```

We recommend installing our package with ROS-supported since RViz visualization is fast and interactive. If you want to use RViz visualization, first run:

```bash
roslaunch nav_gym start_nav_gym.launch
```

and in another terminal, run:

```bash
roslaunch nav_gym view_robot.launch
```

now we can simulate:

```python
import gym
import nav_gym_env
from nav_gym_env.ros_env import RosEnv

env = gym.make("NavGym-v0")
env = RosEnv(env)
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample() # Your agent code here
    obs, reward, done, info = env.step(action)
```

## Credits

Our codebase builds heavily based on [navrep](https://github.com/ethz-asl/navrep) and [
flatland](https://github.com/avidbots/flatland). We appreciate that they have been made open source!


<!-- Note: This reporsitory is part of arena-bench. Please also check out our most recent paper on arena-bench. For our 3D version using Gazebo as simulation platform, please visit our arena-rosnav-3D repo. -->