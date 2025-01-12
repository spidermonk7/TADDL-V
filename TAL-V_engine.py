import numpy as np
# this is a file for loading LLMs
from openai import OpenAI
from argparse import ArgumentParser


# load the args
def load_config():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--key', type=str, default=None, help='OpenAI API key')
    args = arg_parser.parse_args()
    
    return args


args = load_config()


OPENAI_API_KEY = args.key
assert OPENAI_API_KEY is not None, "Please provide the OpenAI API key"


system_prompt_path = 'data/prompts/system_prompt.txt'
task_prompt_path = 'data/prompts/task_prompt.txt'

with open(system_prompt_path, 'r') as f:
    system_prompt = f.read()
    
with open(task_prompt_path, 'r') as f:
    task_prompt = f.read()
    

# load ability masses
ability_masses = np.load('results/FA_GPT4o.npy')

# the ability sets
ability_sets = ['Feature Perception', 'Object Perception', 'Spatial Vision', 'Temporal Vision', 'Visual Reasoning']


client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com/")
task_prompts = "Now you get the inputs:\n" + task_prompt
completion = client.chat.completions.create(
        model = 'deepseek-chat',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompts}
        ]
    )

response = completion.choices[0].message.content
with open(f"results/TAL-V_response.txt", "w") as f:
    f.write(response)


# preprocess the response
try: 
    result = response.split('Required abilities: ')[1].split('\n')[0]
    ability_decomposition = response.split('\n')[2:]

    required_abilities = eval(result)
except:
    ValueError("Some illusion might have happened. Please check the response from the model.")


# Calculate the difficulty level based on the ability masses
difficulty_levels = 0
for ability in required_abilities:
    difficulty_levels += ability_masses[ability - 1]
difficulty_levels /= len(required_abilities)


# show results
required_abilities = [ability_sets[ability - 1] for ability in required_abilities]

# Framework design

def display_task_info(required_abilities, difficulty_levels):
    print("╔═══════════════════════════════════════════╗")
    print("║                TAL-V SYSTEM               ║")
    print("╠════════════════╦══════════════════════════╣")
    print(f"{task_prompt}")
    print("╠════════════════╬══════════════════════════╣")
    print(f"Difficulty level: {difficulty_levels}")
    print("╠════════════════╬══════════════════════════╣")
    print(f"Required abilities:\n {required_abilities}")
    print("╠════════════════╬══════════════════════════╣")
    print(f"Reasons Given by GPT:", ability_decomposition)
    print("╚════════════════╩══════════════════════════╝")

display_task_info(required_abilities, difficulty_levels)
