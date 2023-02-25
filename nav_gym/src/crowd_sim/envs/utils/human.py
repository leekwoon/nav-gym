# -*- coding: utf-8 -*-
import numpy as np

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionRot


class Human(Agent):
    def __init__(self, config, section):
        self.last_state = None
        super().__init__(config, section)

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            action = self.policy.predict(state, local_map, self)
        else:
            action = self.policy.predict(state)

        return action


# 교운 - 2가지 추가기능 지원
# 1) waypoints를 따라 이동
# 2) 사람을 마주하면서 서로 엇갈려 지나갈때 속도를 늦춤
class HumanWithWaypoints(Human):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.robot = None
        # self.slow_around_robot = None 
        self.waypoints = None

    def set_robot(self, robot):
        self.robot = robot

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints

    # def set_slow_around_robot(self, slow_around_robot):
    #     self.slow_around_robot = slow_around_robot

    def is_reach(self):
        pose = np.array([self.px, self.py])
        d = np.linalg.norm(pose - np.array(self.waypoints[-1]))
        return d < 0.1
        # return d < 0.5

    # def act(self, ob, *args, **kwargs):
    #     assert self.waypoints is not None, 'set waypoints'
    #     assert self.robot is not None, 'set robot'
    #     # assert self.slow_around_robot is not None, 'set slow_around_robot'

    #     # find local goal 
    #     # ===============================================
    #     pose = np.array([self.px, self.py])
    #     import time
    #     st = time.time()
    #     while True:
    #         if len(self.waypoints) == 1:
    #             break 

    #         local_goal_distance = np.linalg.norm(pose - np.array(self.waypoints[0]))

    #         if local_goal_distance < 1.:
    #             # self.waypoints.pop(0)
    #             self.waypoints = self.waypoints[1:]
    #         else:
    #             break
    #     self.gx = self.waypoints[0][0]
    #     self.gy = self.waypoints[0][1]
    #     # ===============================================

    #     action = super().act(ob, *args, **kwargs)

    #     return action

    def act(self, ob, *args, **kwargs):
        assert self.waypoints is not None, 'set waypoints'
        assert self.robot is not None, 'set robot'
        # assert self.slow_around_robot is not None, 'set slow_around_robot'

        # find local goal 
        # ===============================================
        pose = np.array([self.px, self.py])
        import time
        st = time.time()
        while True:
            if len(self.waypoints) == 1:
                break 

            local_goal_distance = np.linalg.norm(pose - np.array(self.waypoints[0]))

            if local_goal_distance < 1.:
                # self.waypoints.pop(0)
                self.waypoints = self.waypoints[1:]
            else:
                break
        self.gx = self.waypoints[0][0]
        self.gy = self.waypoints[0][1]
        # ===============================================

        action = super().act(ob, *args, **kwargs)

        # 1. 사람이 로봇을 향해 오는경우 로봇을고려하지 않음. (로봇이 효과적으로 회피하는것을 학습하기 위함)
        # 2. 다만 뒤에서 로봇을 앞지르는경우는 로봇을 고려
        # 사람이 로봇 앞에서 지나가는경우에만 속도를 줄이기 위해
        # 사람이 로봇 앞에 있는지 뒤에있는지 판단
        human_to_humangoal_angle = np.arctan2(
            self.gy - self.py, self.gx - self.px)
        human_to_humangoal_angle = np.arctan2(np.sin(human_to_humangoal_angle), np.cos(human_to_humangoal_angle))
        theta = np.arctan2(np.sin(self.robot.theta), np.cos(self.robot.theta))
        # 로봇의 현재 방향과 비교했을때 사람이 상대적으로 이동할 방향
        heading_1 = np.arctan2(np.sin(theta - human_to_humangoal_angle), np.cos(theta - human_to_humangoal_angle))

        human_angle = np.arctan2(
            self.py - self.robot.py, self.px - self.robot.px)
        human_angle = np.arctan2(np.sin(human_angle), np.cos(human_angle))
        if (-0.5 * np.pi < heading_1 and heading_1 < 0.5 * np.pi):
            self.robot_visible = True
        else:
            self.robot_visible = False
        return action

    # def act(self, ob, *args, **kwargs):
    #     assert self.waypoints is not None, 'set waypoints'
    #     assert self.robot is not None, 'set robot'
    #     # assert self.slow_around_robot is not None, 'set slow_around_robot'

    #     # find local goal 
    #     # ===============================================
    #     pose = np.array([self.px, self.py])
    #     import time
    #     st = time.time()
    #     while True:
    #         if len(self.waypoints) == 1:
    #             break 

    #         local_goal_distance = np.linalg.norm(pose - np.array(self.waypoints[0]))

    #         if local_goal_distance < 1.:
    #             # self.waypoints.pop(0)
    #             self.waypoints = self.waypoints[1:]
    #         else:
    #             break
    #     self.gx = self.waypoints[0][0]
    #     self.gy = self.waypoints[0][1]
    #     # ===============================================

    #     action = super().act(ob, *args, **kwargs)

    #     # 1. 사람이 로봇을 향해 오는경우 로봇을고려하지 않음. (로봇이 효과적으로 회피하는것을 학습하기 위함)
    #     # 2. 다만 뒤에서 로봇을 앞지르는경우는 로봇을 고려
    #     # 사람이 로봇 앞에서 지나가는경우에만 속도를 줄이기 위해
    #     # 사람이 로봇 앞에 있는지 뒤에있는지 판단
    #     human_to_humangoal_angle = np.arctan2(
    #         self.gy - self.py, self.gx - self.px)
    #     human_to_humangoal_angle = np.arctan2(np.sin(human_to_humangoal_angle), np.cos(human_to_humangoal_angle))
    #     theta = np.arctan2(np.sin(self.robot.theta), np.cos(self.robot.theta))
    #     # 로봇의 현재 방향과 비교했을때 사람이 상대적으로 이동할 방향
    #     heading_1 = np.arctan2(np.sin(theta - human_to_humangoal_angle), np.cos(theta - human_to_humangoal_angle))

    #     human_angle = np.arctan2(
    #         self.py - self.robot.py, self.px - self.robot.px)
    #     human_angle = np.arctan2(np.sin(human_angle), np.cos(human_angle))
    #     # 사람이 현재 로봇방향 기준으로 어느 각도에 위치?
    #     heading_2 = np.arctan2(np.sin(self.robot.theta - human_angle), np.cos(self.robot.theta - human_angle))
    #     # 조건1: 사람이 로봇 전방 90 FOV에 위치
    #     # 조건2: 사람이 로봇을 가로질로 이동하는 경우
    #     # -> 속도를 줄임
    #     # if (-0.25 * np.pi < heading_2 and heading_2 < 0.25 * np.pi) \
    #     #     and (-0.5 * np.pi > heading_1 or heading_1 > 0.5 * np.pi):
    #     #     action = ActionRot(min(0.1, action.v), action.r)
    #     if (-0.5 * np.pi < heading_1 and heading_1 < 0.5 * np.pi):
    #         self.robot_visible = True
    #     else:
    #         self.robot_visible = False

    #     # # if too close to robot -> stop and wait
    #     # robot_pose = np.array([self.robot.px, self.robot.py])
    #     # pose = np.array([self.px, self.py])
    #     # dist_to_robot = np.linalg.norm(
    #     #     pose - robot_pose
    #     # )

    #     # 1. 사람이 정면에서 가까운경우 속도를 줄임
    #     # 2. 뒤에서 로봇을 앞질러가는경우는 속도를 줄이지 않음
    #     # if self.robot_visible and self.slow_around_robot:
    #         # if dist_to_robot < self.robot.radius + self.radius + 0.7: # 0.4:
    #         #     # 사람이 로봇 앞에서 지나가는경우에만 속도를 줄이기 위해
    #         #     # 사람이 로봇 앞에 있는지 뒤에있는지 판단
    #         #     human_to_humangoal_angle = np.arctan2(
    #         #         self.gy - self.py, self.gx - self.px)
    #         #     human_to_humangoal_angle = np.arctan2(np.sin(human_to_humangoal_angle), np.cos(human_to_humangoal_angle))
    #         #     theta = np.arctan2(np.sin(self.robot.theta), np.cos(self.robot.theta))
    #         #     # 로봇의 현재 방향과 비교했을때 사람이 상대적으로 이동할 방향
    #         #     heading_1 = np.arctan2(np.sin(theta - human_to_humangoal_angle), np.cos(theta - human_to_humangoal_angle))

    #         #     human_angle = np.arctan2(
    #         #         self.py - self.robot.py, self.px - self.robot.px)
    #         #     human_angle = np.arctan2(np.sin(human_angle), np.cos(human_angle))
    #         #     # 사람이 현재 로봇방향 기준으로 어느 각도에 위치?
    #         #     heading_2 = np.arctan2(np.sin(self.robot.theta - human_angle), np.cos(self.robot.theta - human_angle))
    #         #     # 조건1: 사람이 로봇 전방 90 FOV에 위치
    #         #     # 조건2: 사람이 로봇을 가로질로 이동하는 경우
    #         #     # -> 속도를 줄임
    #         #     # if (-0.25 * np.pi < heading_2 and heading_2 < 0.25 * np.pi) \
    #         #     #     and (-0.5 * np.pi > heading_1 or heading_1 > 0.5 * np.pi):
    #         #     #     action = ActionRot(min(0.1, action.v), action.r)
    #     return action

