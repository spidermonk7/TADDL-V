import os
import openai
import pandas as pd

# 设置 API 密钥和相关配置
openai.api_key = "sk-MSSwI7MgizQFSyUE64359c5000D64b518cCc7c00F30e0321"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}

if __name__ == '__main__':
    # 加载系统提示词和任务提示词
    system_prompt_path = "./data/prompts/system_prompt.txt"


    # 读取现有的 Excel 文件
    excel_path = './data/source/AK_marked_v4.xlsx'
    df = pd.read_excel(excel_path)

    # 确保新列存在
    if 'Required Abilities(modified)' not in df.columns:
        df['Required Abilities(modified)'] = ''

    if 'Explanation' not in df.columns:
        df['Ability Explanations'] = ''


    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()

    for i in range(70):
        print(f"Processing task {i}")
        task_prompt_path = f"./data/prompts/task_prompts/task_prompt_{i}.txt"
        with open(task_prompt_path, 'r') as f:
            task_prompts = f.read()
        prompt = system_prompt + task_prompts
        completion = openai.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": 'user',
                    "content": prompt,
                }
            ]
        )
        required_abilities = completion.choices[0].message.content
        explanation = required_abilities.split("[")[0]

        abilities = required_abilities.split("[")[-1].strip("*")[:-1]
        df.loc[i, 'Ability Explanations'] = explanation
        df.loc[i, 'Required Abilities(modified)'] = abilities

    df.to_excel(excel_path, index=False)
