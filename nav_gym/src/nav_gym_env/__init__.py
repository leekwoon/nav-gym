from gym.envs.registration import register


register(
    id='NavGym-v0',
    kwargs={
        'robot_type': 'keti',
        'time_step': 0.2,
        'min_turning_radius': 0, # 0.33712, # wheel_base / 2
        'distance_threshold': 0.5, # less then it -> reach goal
        'num_scan_stack': 1,
        'linvel_range': [0, 0.5],
        'rotvel_range': [-0.64, 0.64],
        'human_v_pref_range': [0., 0.6],
        'human_has_legs_ratio': 0.5, # modeling human leg movement
        'indoor_ratio': 0.5,
        'min_goal_dist': 10, 
        'max_goal_dist': 20, 
        'reward_scale': 15.,
        'reward_success_factor': 1,
        'reward_crash_factor': 1,
        'reward_progress_factor': 0.001,
        'reward_forward_factor': 0.0,
        'reward_rotation_factor': 0.005,
        'reward_discomfort_factor': 0.01,
        'env_param_range': dict(
            # num_humans=([10, 15], 'int'), 
            # num_humans=([5, 35], 'int'), 
            num_humans=([5, 15], 'int'), 
            # indoor map param
            corridor_width=([3, 4], 'int'), 
            iterations=([80, 150], 'int'), 
            # outdoor map param
            obstacle_number=([10, 10], 'int'), 
            obstacle_width=([0.3, 1.0], 'float'), 
            scan_noise_std=([0., 0.05], 'float'), 
        ),
    },
    entry_point='nav_gym_env.env:NavGymEnv'
)