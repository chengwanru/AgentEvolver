
from crafter_env.create_env import make_crafter_env

config = {
    "crafter_kwargs": {
        "area": [64, 64],
        "view": [9, 9],
        "size": [256, 256],
        "reward": True,
        "seed": None,
        "max_episode_steps": 2000,
        "unique_items": True,
        "precise_location": False,
        "skip_items": [],             # 示例: ["grass", "sand", "path"]
        "edge_only_items": [],        # 示例: ["water", "lava"]
    }
}


env = make_crafter_env(config)
obs, info = env.reset()

sts=env.get_stats()

import random
def get_query_list(split: str = "train"):
    # 根据 split 决定采样数量
    if split == 'test':
        num_samples = 10
    elif split == 'val':
        num_samples = 20
    elif split == 'train':
        num_samples = 100
    else:
        raise ValueError(f'Unknown split: {split}')

    # 创建本地random对象，保证独立性
    rng = random.Random(1995)
    max_num = 19950807

    # 采样
    goal_idxs = rng.sample(range(max_num), num_samples)
    return goal_idxs

lists_eric =get_query_list('train')

print(env)

