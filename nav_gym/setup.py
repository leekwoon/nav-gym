from setuptools import setup


setup(
    name='nav_gym',
    description='navigation environment - Gym',
    author='Kyowoon Lee',
    version='1.0',
    packages=['crowd_nav', 'crowd_sim', 'nav_gym_env', 'nav_gym_ros'],
    package_dir={'':'src'},
    install_requires=[
        'matplotlib',
        'pandas==1.1.0',
        'rospkg',
        'netifaces',
        'gym==0.10.5',
        'opencv-python==4.6.0.66',
        'PyQt5==5.9.2',
        'torch==1.4.0', # dynamic obstacles
        # 3rd party packages
        'pyastar2d', # to generate waypoints fast
        'pymap2d',
        'pyrangelibc-danieldugas',
        'pyrvo2-danieldugas'
    ]
)