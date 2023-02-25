# nav-env

A OpenAI-gym compatible navigation simulator, which can be integrated into the robot operating system (ROS) with the goal of open-sourcing it for easy comparison of various approaches including state- of-the-art learning-based approaches and conventional ones.

## Install

The package has been tested on Ubuntu 18.04 / Python 3.6 / ROS Melodic.

To install without ROS-supported:  

```bash
virtualenv ~/venv/hrlnav --python=python3.6
source ~/venv/hrlnav/bin/activate

git clone https://github.com/leekwoon/nav_gym.git

cd nav_gym/nav_gym
pip install -e .
```

To install with ROS-supported:  

```bash
virtualenv ~/venv/hrlnav --python=python3.6
source ~/venv/hrlnav/bin/activate

cd ~/YOUR_CATKIN_WORKSPACE/src
git clone https://github.com/leekwoon/nav_gym.git

# install nav_gym_env
cd nav_gym/nav_gym
pip install -e .

cd ~/YOUR_CATKIN_WORKSPACE
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash # or you can write it on ~/.bashrc
```

<!-- Note: This reporsitory is part of arena-bench. Please also check out our most recent paper on arena-bench. For our 3D version using Gazebo as simulation platform, please visit our arena-rosnav-3D repo. -->