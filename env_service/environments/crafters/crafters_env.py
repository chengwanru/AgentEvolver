# -*- coding: utf-8 -*-

from typing import Dict, List
import random


from EnvService.env_sandbox.base import BaseEnv
from EnvService.env_sandbox.registry import Registry


from EnvService.env_sandbox.environments.crafters.crafters.utils import get_instruction_prompt
from EnvService.env_sandbox.environments.crafters.crafters.create_env import make_crafter_env
#from EnvService.env_sandbox.environments.barlog.utils import HistoryPromptBuilder

naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()
sys_prompt = get_instruction_prompt() +"\n\n" + naive_instruction

@Registry.register("crafters")
class CraftersEnv(BaseEnv):
    def __init__(
            self,
            task_id: str = None,
            instance_id: str = None,
            params: Dict = None,
    ):

        self.task_id = int(task_id) if task_id is not None else None
        self.instance_id = instance_id
        self.env = None
        params = params or {}
        self.failed_candidates=[]

        self.reward=0.0

        self.config ={
    "crafter_kwargs": {
        "area": params['area'] if isinstance(params.get('area'), list) else [64, 64],
        "view": params['view'] if isinstance(params.get('view'), list) else [9, 9],
        "size": params['size'] if isinstance(params.get('size'), list) else [256, 256],
        "reward": True,
        "seed": self.task_id,
        "max_episode_steps": params['max_step'] if isinstance(params.get('max_step'), int) else 2000,
        "unique_items": params['unique_items'] if isinstance(params.get('unique_items'), bool) else True,
        "precise_location": params['precise_location'] if isinstance(params.get('precise_location'), bool) else False,
        "skip_items": params['skip_items'] if isinstance(params.get('skip_items'), list) else [],
        "edge_only_items": params['edge_only_items'] if isinstance(params.get('edge_only_items'), list) else [],
    }
        }

        # self.prompt_builder=HistoryPromptBuilder(
        #     max_text_history=params['max_history'] if isinstance(params.get('max_history'), int) else 16,
        #     max_image_history=0, #remove VL
        #     max_cot_history=0,# use naive agent
        # )

        self.last_obs=None





    def get_init_state(self, params: Dict = None):


        self.env =  make_crafter_env(self.config)
        obs, info = self.env.reset()

        self.last_obs=obs["text"].get("short_term_context", "")

        #todo: fixed the current obs & obs
        text_obs=obs["text"].get("long_term_context", "") + "Observation:" + self.last_obs


        return {
            "state": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text_obs},
            ],
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }


    def step(self, action: Dict, params: Dict = None):
        params = params or {}

        action_msg = action['content']

        success,action = self.check_action_validity(action_msg)

        if success:
            text_obs=""
        else:
            text_obs = f"\n\nYour previous output {action_msg} did not contain a valid action. Defaulted to action: {action}\n\nObservation:\n"


        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward+=reward

        is_terminated = terminated or truncated



        self.last_obs = obs["text"].get("short_term_context", "")

        # todo: fixed the current obs & obs
        text_obs += obs["text"].get("long_term_context", "") + "Observation:" + self.last_obs

        return {
            "state": [{"role": "user", "content": text_obs}],
            "reward": reward,
            "is_terminated": is_terminated,
            "info": {
                'progression':self.get_stats()['progression'],
                'sum_reward':self.reward
            },
        }


    def evaluate(self, messages: Dict = None, params: Dict = None) -> float:

        if self.env is not None:
            return self.get_stats()['score']/100.0
        else:
            return 0.0


    def get_info(self):
        if self.env:
            current_status = self.get_stats()

            return "Observation:" + self.last_obs + f"progression : {current_status['progression']} and score is {current_status['score']}"
        else:
            return ""

    def close(self):
        if self.env:
            self.env.close()


    def check_action_validity(self, candidate_action):
        valid_action = None
        success=False
        if candidate_action in self.env.language_action_space:
            valid_action = candidate_action
            success = True
        else:
            valid_action = self.env.default_action
            self.failed_candidates.append(candidate_action)
            success = False
        return success,valid_action

    def get_stats(self):
        return self.env.get_stats()



    @staticmethod
    def get_query_list(split: str = "train",params={}):

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
        max_num=19950807

        # 采样
        goal_idxs = rng.sample(range(max_num), num_samples)
        return goal_idxs


if __name__ == '__main__':
    envs=CraftersEnv(task_id=1003)
    msg=envs.get_init_state()
    sc=envs.get_stats()
    print(msg)
