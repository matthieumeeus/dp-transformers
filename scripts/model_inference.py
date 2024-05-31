#%%
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model_name_or_path = "phi_2_dpo_orca_dpo_pairs"

#%%
if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
    from peft import PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(model_name_or_path)
    base_model_path = config.base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

#%%
prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant
 gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative 
 ideas for a 10 year old's birthday?\n### Assistant: Of course! Here are some creative ideas for 
 a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. 
 Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. 
 Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. 
 You can set up different stations with activities like making slime, erupting volcanoes, or creating 
 simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector 
 and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve 
 popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange 
 a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, 
 paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. 
 Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for 
 activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or 
 certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare 
 their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let 
 them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids 
 can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes 
 or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure 
 party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with 
 games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday 
 child's interests and preferences. Have a great celebration!\n### Human: Evaluate the following movie reviews on a 
 scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:\n1. This movie released on 
 Nov. 18, 2019, was phenomenal. The cinematography, the acting, the plot - everything was top-notch.\n2. Never before 
 have I been so disappointed with a movie. The plot was predictable and the characters were one-dimensional. In my opinion, 
 this movie is the worst one to have been released in 2022.\n3. The movie was okay. There were some parts I  enjoyed, but 
 there were also parts that felt lackluster. This is a movie that was released in Feb 2018 and seems to be quite 
 ordinary.\nReturn the answer as a JSON array of integers.\n### Assistant:"""

#%%
prompt = """Instruct: Evaluate the following movie reviews on a 
 scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:\n1. This movie released on 
 Nov. 18, 2019, was phenomenal. The cinematography, the acting, the plot - everything was top-notch.\n2. Never before 
 have I been so disappointed with a movie. The plot was predictable and the characters were one-dimensional. In my opinion, 
 this movie is the worst one to have been released in 2022.\n3. The movie was okay. There were some parts I  enjoyed, but 
 there were also parts that felt lackluster. This is a movie that was released in Feb 2018 and seems to be quite 
 ordinary.\nReturn the answer as a JSON array of integers.\nOutput:"""

#%%
prompt = """\nAlice: Compose an engaging travel blog post about a recent 
trip to Hawaii, highlighting cultural experiences and must-see 
attractions.\nBob:"""

#%%
prompt = """<|system|>\n<|endoftext|>\n<|user|>\nDescribe five key 
principles in evaluating an argument in analytical 
writing.<|endoftext|>\n<|assistant|>\n"""

#%%
input_ids = tokenizer([prompt]).input_ids

do_sample = False
temperature = 0.0
max_new_token = 1024

## Don't return the prompt
output_ids = model.generate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_token,
                eos_token_id=tokenizer.eos_token_id,                
            )
output_ids = output_ids[0][len(input_ids[0]):]

output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )

print(output)
# %%
# USE THIS TO FIRST LOAD THE INSTRUCTION MODEL AND THEN THE DPO MODEL
model_name_or_path = "phi_2_instruction"
from peft import PeftConfig, PeftModel
config = PeftConfig.from_pretrained(model_name_or_path)
base_model_path = config.base_model_name_or_path
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, trust_remote_code=True)
instruction_model = PeftModel.from_pretrained(base_model, model_name_or_path)
instruction_model.eval()
instruction_model = instruction_model.merge_and_unload()

model_name_or_path = "phi_2_dpo_orca_dpo_pairs"
config = PeftConfig.from_pretrained(model_name_or_path)
model = PeftModel.from_pretrained(instruction_model, model_name_or_path)
model.eval()
model = model.merge_and_unload()
# %%
