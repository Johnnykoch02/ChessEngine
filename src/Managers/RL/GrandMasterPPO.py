from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class GrandMasterPPO(PPO):
    '''
        Goal: Implement custom RL Training algorithm to target the game of chess through the GrandMasterEnv.
        Implementation:
            1. https://github.com/DLR-RM/stable-baselines3/blob/002850f8ace0e045f7e9d370149a6fbb6cbcebad/stable_baselines3/common/on_policy_algorithm.py#L20
            2. https://github.com/DLR-RM/stable-baselines3/blob/002850f8ace0e045f7e9d370149a6fbb6cbcebad/stable_baselines3/common/base_class.py#L367
        Github PPO Details the Paper Trail of how to accomplish this.
        - Needs:
            - Model Controllers: two models, 1 for each color, Model Controller will need to implement two GrandMasterPPOs and
            train after N number of steps, similar to how their algorithms already function. 
            - These Models are going to share a common enviornment
            - 
    '''