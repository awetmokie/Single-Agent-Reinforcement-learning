import gym
import highway_env
import numpy as np
from actor_critic_torch import Agent
from utils import plot_learning_curve
import math
import warnings
warnings.filterwarnings('ignore')


def check_observation(observation):

    distance = []

    for i in range(1, len(observation)):
        distance.append(math.sqrt((observation[i][1])**2 ))
    # distance = math.sqrt((observation[1][1])**2 + (( observation[1][2])**2))
    return np.mean(distance)

if __name__ == '__main__':
    env = gym.make('joker-highway-v0')
    # env.configure({
    # "action": {
    #     "type": "DiscreteMetaAction"
    #    }
    # })

    obs =env.reset()
    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.n
    input_dims = env.observation_space.shape[1]*env.observation_space.shape[0]

    print(num_inputs)
    print(num_outputs)

    # highway_env lr = 5e-4

    agent = Agent(gamma=0.99, lr=5e-4, input_dims=[input_dims], n_actions= num_outputs,
                  fc1_dims= 64, fc2_dims= 64)
    n_games = 3000

    fname = 'CC_ACTOR_CRITIC_version_2' + 'joker-highway-v0' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        
        while not done:
            # env.render(mode ="Human")
            action,_ = agent.choose_action(observation.reshape(input_dims))
            #print("---------------------- action ----------------------")
            #print(action)
            observation_, reward, done, info = env.step(action)
            cost =check_observation(observation) - check_observation(observation)* int(info['agents_costs'][0])
            # print(cost)
            score += reward

            agent.learn(observation.reshape(input_dims), reward, observation_.reshape(input_dims), done,int(info['agents_costs'][0]))
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    
    agent.save_model(  path="state_dict_model_v2.pt")

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)

