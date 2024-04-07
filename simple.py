
import gym
import highway_env
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib
import random
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    env = gym.make('highway-v0')     
   
    # simple environment configuration
    env.configure({ 
    "action": {
        "type": "DiscreteMetaAction"
       }
    })

    n_games = 30    # number of games
    

    actions = [0,1,2,3,4]  # action list  

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        
        while not done:
            env.render(mode ="Human")  #
            observation_, reward, done, info = env.step(random.choice(actions))   # chosing random action from action list 




