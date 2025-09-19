from env_service.env_client import EnvClient


def agent_test(task_id=0):
    client = EnvClient(base_url="http://localhost:8080")

    # 获取任务列表
    env_type = "crafters"

    task_ids = client.get_env_profile(env_type, split='train')

    print(task_ids)

    init_response = client.create_instance(env_type,
                                           task_ids[task_id],
                                           params={
                                               'max_step': 40,  # prevent in case
                                               'area': [64, 64],
                                               "size": [256, 256]
                                           }
                                           )
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # system_msgs = init_response['info']['system_prompt']
    # messages_list=[
    #     {"role": "system", "content": system_msgs},
    # ]
    messages_list = []

    finish = False
    move=0
    while not finish:
        move+=1

        user_msgs = init_response['state']
        # print('input state is ' + user_msgs)
        messages_list.extend(user_msgs)
        # print('----')
        # print(messages_list)
        # print('----')

        from openai import OpenAI

        sse_client = OpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model_name = "qwen-max"  # Always use GPT-4o regardless of input model_name

        response = sse_client.chat.completions.create(
            model=model_name,
            messages=messages_list,
            # messages=[
            #     {"role": "system", "content": system_msgs},
            #     {"role": "user", "content": user_msgs},
            # ],
        )
        action = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        print(f'in the move :{move}')
        print(action)
        messages_list.append(action)

        print('----steping----')

        result = client.step(instance_id, action)
        print(f"Step result: {result}")

        finish = result['is_terminated']

        init_response = result

    print(result['reward'])

    return result['reward']

    # 使用示例


def main():
    client = EnvClient(base_url="http://localhost:8080")

    # 获取任务列表
    env_type = "webshop"
    task_ids = client.get_env_profile(env_type, split='train')
    print(f"Available tasks: {task_ids}")

    # 创建实例
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id, params={
        'base_url': 'http://127.0.0.1:1907/',  # prevent in case
        'human_goals': False,

    })

    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # 获取环境信息

    tool_info = client.get_tools_info(instance_id, params={"prompt": False})
    print(f"env information is  {instance_id} : {tool_info}")

    # 执行动作
    # "对于appworld数据集，需要以str的方式返回action，str中需包含```python```代码块"

    action = {
        "role": "assistant",
        "content": """\\boxed{search[machine wash men's dress shirts cotton spandex classic fit short sleeve color coral sands size 3x-large tall price < 60.00]}""",
    }

    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # 释放实例
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")

    # todo reset


if __name__ == "__main__":
    agent_test()
