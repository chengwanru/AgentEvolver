# -*- coding: utf-8 -*-
"""AvalonTrainer class for training agents in Avalon game."""
import os
import json
import yaml
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

from games.avalon.utils import GameLogger

@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    name: str
    api_key: str
    temperature: float = 0.7
    stream: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            name=config_dict.get('name', 'qwen-plus'),
            api_key=config_dict.get('api_key', os.getenv('API_KEY', '')),
            temperature=config_dict.get('temperature', 0.7),
            stream=config_dict.get('stream', True),
        )


@dataclass
class TaskConfig:
    """Task configuration dataclass."""
    num_players: int = 5
    language: str = 'en'
    log_dir: str = 'logs'
    default_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name='qwen-plus',
        api_key=os.getenv('API_KEY', ''),
    ))
    train_roles: set = field(default_factory=set)
    no_train_roles: set = field(default_factory=set)
    custom_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfig':
        game_config = config_dict.get('game', {})
        roles_config = config_dict.get('roles', {})
        return cls(
            num_players=game_config.get('num_players', 5),
            language=game_config.get('language', 'en'),
            log_dir=game_config.get('log_dir', 'logs'),
            default_model=ModelConfig.from_dict(config_dict.get('default_model', {})),
            train_roles={r.lower() for r in roles_config.get('train', [])},
            no_train_roles={r.lower() for r in roles_config.get('no_train', [])},
            custom_configs={k.lower(): v for k, v in roles_config.get('custom_configs', {}).items()},
        )


class RoleManager:
    """Manages role indexing and identification."""
    
    def __init__(self, roles: List[Tuple[int, str, bool]]):
        self.roles = roles
        role_counters = defaultdict(int)
        self.indexed_roles = []
        for _, role_name, _ in roles:
            self.indexed_roles.append(f"{role_name}_{role_counters[role_name]}")
            role_counters[role_name] += 1
    
    def get_indexed_role(self, index: int) -> str:
        return self.indexed_roles[index]
    
    def get_role_name(self, index: int) -> str:
        return self.roles[index][1]
    
    def is_good(self, index: int) -> bool:
        return self.roles[index][2]


class AvalonTrainer:
    """Trainer class for Avalon game that runs games and collects training data."""
    
    def __init__(self, task_config: Union[str, Dict[str, Any]], model_path: str):
        config_dict = self._load_config(task_config)
        self.task_config = TaskConfig.from_dict(config_dict)
        self.model_path = model_path
    
    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return config
    
    @staticmethod
    def _create_result_dir(base_dir: str = 'results', timestamp: str = None) -> str:
        """Create result directory with timestamp. Reuses logic from GameLogger."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(base_dir, f'game_{timestamp}')
        os.makedirs(result_dir, exist_ok=True)
        return result_dir
    
    def _is_training_role(self, indexed_role: str, base_role: str) -> bool:
        return (indexed_role.lower() in self.task_config.train_roles or
                base_role.lower() in self.task_config.train_roles)
    
    def _get_model_config(self, indexed_role: str, base_role: str, is_training: bool) -> ModelConfig:
        config_dict = {
            'name': self.model_path if is_training else self.task_config.default_model.name,
            'api_key': self.task_config.default_model.api_key,
            'temperature': self.task_config.default_model.temperature,
            'stream': self.task_config.default_model.stream,
        }
        for role_key in [indexed_role.lower(), base_role.lower()]:
            if role_key in self.task_config.custom_configs:
                config_dict.update(self.task_config.custom_configs[role_key])
                break
        return ModelConfig(**config_dict)
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        from agentscope.model import DashScopeChatModel
        from agentscope.formatter import DashScopeMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        
        model_config = self._get_model_config(indexed_role, base_role, 
                                             self._is_training_role(indexed_role, base_role))
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=DashScopeChatModel(
                model_name=model_config.name,
                api_key=model_config.api_key,
                stream=model_config.stream,
            ),
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
    
    def _create_agents(self, role_manager: RoleManager) -> List:
        return [
            self._create_agent(i, role_manager.get_indexed_role(i), role_manager.get_role_name(i))
            for i in range(len(role_manager.roles))
        ]
    
    def _identify_training_agents(self, agents: List, role_manager: RoleManager) -> List[int]:
        training_indices = []
        for i, agent in enumerate(agents):
            indexed_role = role_manager.get_indexed_role(i)
            if not self._is_training_role(indexed_role, role_manager.get_role_name(i)):
                continue
            if hasattr(agent, 'model') and hasattr(agent.model, 'model_name'):
                if agent.model.model_name != self.model_path:
                    raise ValueError(
                        f"Agent {i} ({indexed_role}) has model '{agent.model.model_name}' "
                        f"but expected '{self.model_path}'"
                    )
            training_indices.append(i)
        
        if not training_indices:
            raise ValueError(
                f"No training agents found. Train roles: {self.task_config.train_roles}, "
                f"Assigned roles: {role_manager.indexed_roles}"
            )
        return training_indices
    
    def _collect_training_data(
        self, agents: List, training_indices: List[int],
        role_manager: RoleManager, good_victory: bool,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Collect training data from agents and save to files."""
        # Create result directory using helper method
        game_dir = self._create_result_dir('results', timestamp)
        
        all_histories = []
        
        # Convert good_victory to Python bool using GameLogger's utility
        good_victory_bool = GameLogger._convert_to_serializable(good_victory)
        
        for idx in training_indices:
            indexed_role = role_manager.get_indexed_role(idx)
            # Use GameLogger's utility to convert numpy types
            is_good = GameLogger._convert_to_serializable(role_manager.is_good(idx))
            reward = 1.0 if is_good == good_victory_bool else 0.0
            model_call_history = getattr(agents[idx], 'model_call_history', [])
            num_model_calls = len(model_call_history)
            all_histories.extend(model_call_history)
            
            # Prepare data and convert numpy types
            agent_data = {
                'agent_info': {
                    'player_id': idx,
                    'indexed_role': indexed_role,
                    'base_role': role_manager.get_role_name(idx),
                    'is_good': is_good,
                },
                'reward': reward,
                'num_model_calls': num_model_calls,
                'model_call_history': GameLogger._convert_to_serializable(model_call_history),
            }
            
            # Save individual JSON file for each training role
            file_path = os.path.join(game_dir, f'{indexed_role}_history.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(agent_data, f, ensure_ascii=False, indent=2)
        
        return {
            'model_call_history': GameLogger._convert_to_serializable(all_histories),
            'game_dir': game_dir,
        }
    
    async def train(self) -> Dict[str, Any]:
        """Run a training game and collect training data."""
        from games.avalon.game import AvalonGame
        from games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
        
        # Generate timestamp once for both logs and results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        config = AvalonBasicConfig.from_num_players(self.task_config.num_players)
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
        role_manager = RoleManager(assigned_roles)
        agents = self._create_agents(role_manager)
        training_indices = self._identify_training_agents(agents, role_manager)
        
        game = AvalonGame(
            agents=agents,
            config=config,
            log_dir=self.task_config.log_dir,
            language=self.task_config.language,
            preset_roles=assigned_roles,
            timestamp=timestamp,
        )
        good_victory = await game.run()
        return self._collect_training_data(agents, training_indices, role_manager, good_victory, timestamp)
