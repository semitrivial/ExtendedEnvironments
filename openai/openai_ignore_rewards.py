import openai

engine="ada"

with open("key.dat", "r") as fp:
  openai.api_key = fp.readline()[:-1]

with open("agent.txt", "r") as fp:
  base_prompt = fp.read()

base_prompt += "\nNew environment\n"

true_prompt = base_prompt
oracle_prompt = base_prompt

bias = {
  198: -100,   # '\n'
  3791: -100   # 'New'
}

def query(prompt):
  x = openai.Completion.create(engine=engine, prompt=prompt, n=5, max_tokens=2, temperature=0.5, logit_bias=bias)
  for i in range(5):
    choice = x.choices[i].text
    if choice == ">A":
      return "A"
    if choice == ">B":
      return "B"
  import pdb; pdb.set_trace()

for _ in range(15):
  oracle_response = query(oracle_prompt)
  oracle_prompt += '>' + oracle_response + '\nOk\n'
  true_response = query(true_prompt)
  true_prompt += '>' + true_response + '\n'
  if true_response == oracle_response:
    true_prompt += 'Good\n'
  else:
    true_prompt += 'Bad\n'

with open(engine+"_result.txt", "w") as fp:
  fp.write(true_prompt)
with open(engine+"_oracle.txt", "w") as fp:
  fp.write(oracle_prompt)

import pdb; pdb.set_trace()

#response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=5, temperature=0.5)
