# -*- coding: utf-8 -*-
"""Test script for AvalonTrainer."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from games.avalon.trainer import AvalonTrainer


async def test_trainer(task_config_path: str, model_path: str):
    """Test AvalonTrainer by running a training game and collecting data."""
    print("=" * 80)
    print("Testing AvalonTrainer")
    print(f"Task config: {task_config_path}")
    print(f"Model path: {model_path}")
    print("=" * 80)
    
    trainer = AvalonTrainer(task_config=task_config_path, model_path=model_path)
    print(f"\nTrainer: {trainer.task_config.num_players} players, "
          f"train roles: {trainer.task_config.train_roles}, "
          f"model: {model_path}")
    
    print("\nStarting training game...")
    result = await trainer.train()
    
    print(f"\nResults: Reward={result['reward']}, "
          f"Model calls={len(result['model_call_history'])}")
    print("Training data saved to results/{role_name}_history.jsonl files")
    print("Training completed!")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AvalonTrainer")
    parser.add_argument("--task-config", type=str, default="games/avalon/task_config.yaml",
                       help="Path to task config YAML file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to be trained")
    
    args = parser.parse_args()
    
    # Resolve task config path
    task_config_path = args.task_config
    if not os.path.isabs(task_config_path):
        root = Path(__file__).parent.parent.parent
        for path in [root / task_config_path, Path.cwd() / task_config_path, Path(__file__).parent / task_config_path]:
            if path.exists():
                task_config_path = str(path)
                break
    
    if not os.path.exists(task_config_path):
        print(f"Warning: Task config file not found at: {task_config_path}")
    
    asyncio.run(test_trainer(task_config_path, args.model_path))
