from typing import Optional

import crafter
from EnvService.env_sandbox.environments.crafters.crafters.utils import CrafterLanguageWrapper
from EnvService.env_sandbox.environments.crafters.gym_compatibility import GymV21CompatibilityV0


def make_crafter_env( config, render_mode: Optional[str] = None):
    crafter_kwargs = dict(config["crafter_kwargs"])
    max_episode_steps = crafter_kwargs.pop("max_episode_steps", 2)
    unique_items = crafter_kwargs.pop("unique_items", True)
    precise_location = crafter_kwargs.pop("precise_location", False)
    skip_items = crafter_kwargs.pop("skip_items", [])
    edge_only_items = crafter_kwargs.pop("edge_only_items", [])

    for param in ["area", "view", "size"]:
        if param in crafter_kwargs:
            crafter_kwargs[param] = tuple(crafter_kwargs[param])

    crafter_kwargs['length'] = max_episode_steps


    env = crafter.Env(**crafter_kwargs)
    env = CrafterLanguageWrapper(
        env,
        task="",
        max_episode_steps=max_episode_steps,
        unique_items=unique_items,
        precise_location=precise_location,
        skip_items=skip_items,
        edge_only_items=edge_only_items,
    )
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
