from env_service.env_client import EnvClient



def agent_test(task_id=0):



    client = EnvClient(base_url="http://localhost:8080")

    # 获取任务列表
    env_type = "openworld"

    task_ids = client.get_env_profile(env_type, split='train')

    init_response = client.create_instance(env_type,
                                           str(task_ids[task_id]),
                                           )
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # system_msgs = init_response['info']['system_prompt']
    # messages_list=[
    #     {"role": "system", "content": system_msgs},
    # ]
    messages_list =[]


    finish = False
    while not finish:

        user_msgs = init_response['state']
        #print('input state is ' + user_msgs)
        messages_list.extend(user_msgs)
        print('----')
        print(messages_list)
        print('----')


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
        print(action)
        messages_list.append(action)
        #for debug use to check the action


        print('----steping----')

        result = client.step(instance_id, action)
        print(f"Step result: {result}")

        finish = result['is_terminated']

        init_response=result

    print(result['reward'])

    return result['reward']





if __name__ == "__main__":
    agent_test()
