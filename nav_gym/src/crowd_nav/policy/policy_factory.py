from crowd_nav.policy.cadrl_original_data import CADRL_ORIGINAL
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.policy.random_policy import RandomPolicy

policy_factory = dict()

policy_factory['cadrl_original'] = CADRL_ORIGINAL
policy_factory['sarl'] = SARL
policy_factory['orca'] = ORCA
policy_factory['sdoadrl'] = SDOADRL
policy_factory['random'] = RandomPolicy
