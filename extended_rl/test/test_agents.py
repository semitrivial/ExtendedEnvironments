# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.

from random import random

def zero_agent(prompt, *meta):
    return 0

def random_agent(prompt, *meta):
    return 0 if random() < .5 else 1

def alternating_agent(prompt, *meta):
    return 0 if (len(prompt)%2)==0 else 1

def obs_repeating_agent(prompt, *meta):
    return prompt[-1]

agents = {
    'zero_agent': zero_agent,
    'random_agent': random_agent,
    'alternating_agent': alternating_agent,
    'obs_repeating_agent': obs_repeating_agent,
}