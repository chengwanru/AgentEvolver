需要按照Python 版本是

```{code-cell}
conda create -n balrog python=3.10 -y
```

请注意，单次任务的 max step 数量可以通过下面的创建instance的时候指定

```{code-cell}
 init_response = client.create_instance(env_type,
                                           task_ids[task_id],
                                           params={
                                               'max_step': 40,  # prevent in case
                                               'area': [64, 64],
                                               "size": [256, 256]
                                           }
                                           )
```                

也可以通过修改主文件`crafters_env.py` 中`init`里的参数设置进行修改

```{code-cell}
 "max_episode_steps": params['max_step'] if isinstance(params.get('max_step'), int) else 2000,
```                

这个游戏总共有2个评价方式，一个是完成度，一个是评分。目前使用完成度进行评价
在每一个step过程中，会产生单个action的reward（可能为负、可能为0 、可能为1） ，也记录了完整的累积reward
下面是一个step的返回结果
```{code-cell}
{'state': [{'role': 'user', 'content': '\n\nYour previous output Move North-East did not contain a valid action. Defaulted to action: Noop\n\nObservation:\nYou see:\n- grass 1 step to your west\n- tree 2 steps to your north-west\n- cow 3 steps to your north-east\n\nYou face grass at your front.Observation:Your status:\n- health: 9/9\n- food: 8/9\n- drink: 8/9\n- energy: 8/9\n\nYour inventory:\n- wood: 2'}], 
'reward': 0.0, 'is_terminated': False, 'info': {'progression': 0.045454545454545456, 'sum_reward': 1.0}}
```                

可以执行的动作指令放在system prompt中，通过init 拿到，例如
```{code-cell}
msg=envs.get_init_state()
{'state': [{'role': 'system', 'content': 'You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:\n\nNoop: do nothing,\nMove West: move west on flat ground,\nMove East: move east on flat ground,\nMove North: move north on flat ground,\nMove South: move south on flat ground,\nDo: Multiuse action to collect material, drink from lake and hit creature in front,\nSleep: sleep when energy level is below maximum,\nPlace Stone: place a stone in front,\nPlace Table: place a table,\nPlace Furnace: place a furnace,\nPlace Plant: place a plant,\nMake Wood Pickaxe: craft a wood pickaxe with a nearby table and wood in inventory,\nMake Stone Pickaxe: craft a stone pickaxe with a nearby table, wood, and stone in inventory,\nMake Iron Pickaxe: craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory,\nMake Wood Sword: craft a wood sword with a nearby table and wood in inventory,\nMake Stone Sword: craft a stone sword with a nearby table, wood, and stone in inventory,\nMake Iron Sword: craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory.\n\nThese are the game achievements you can get:\n1. Collect Wood\n2. Place Table\n3. Eat Cow\n4. Collect Sapling\n5. Collect Drink\n6. Make Wood Pickaxe\n7. Make Wood Sword\n8. Place Plant\n9. Defeat Zombie\n10. Collect Stone\n11. Place Stone\n12. Eat Plant\n13. Defeat Skeleton\n14. Make Stone Pickaxe\n15. Make Stone Sword\n16. Wake Up\n17. Place Furnace\n18. Collect Coal\n19. Collect Iron\n20. Make Iron Pickaxe\n21. Make Iron Sword\n22. Collect Diamond\n\nIn a moment I will present a history of actions and observations from the game.\nYour goal is to get as far as possible by completing all the achievements.\n\nPLAY!\n\nYou always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.'}, 
{'role': 'user', 'content': 'You see:\n- water 7 steps to your north-west\n- grass 1 step to your west\n- cow 5 steps to your south-west\n- tree 5 steps to your north-east\n\nYou face grass at your front.Observation:Your status:\n- health: 9/9\n- food: 9/9\n- drink: 9/9\n- energy: 9/9\n\nYou have nothing in your inventory.'}], 
'info': {'instance_id': None, 'task_id': 1003}}

```                
在barlog代码中，它把system prompt塞入了user prompt且每个step都会给，由于ba会保存system prompt所以目前就这么处理了


训练和测试和验证用的是不同种子生成的地图，具体设置目前在`crafters_env.py` 中，目前的数量是
        # 根据 split 决定采样数量
        if split == 'test':
            num_samples = 10
        elif split == 'val':
            num_samples = 20
        elif split == 'train':
            num_samples = 100

也可以根据不同的step 长度进行修改，也需要对环境clone进行支持，避免总是学初始化状态

可以在主目录下的test_crafter 进行测试