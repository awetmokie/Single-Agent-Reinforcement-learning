import gym
import highway_env
import numpy as np
from actor_critic_torch import Agent
from utils import plot_learning_curve
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
   

    env.reset()
    #input_dims = env.observation_space.shape[1]*env.observation_space.shape[0]
    agent = Agent(gamma=0.99, lr=5e-5, input_dims=[num_inputs], n_actions= num_outputs,
                  fc1_dims=256, fc2_dims=256)
    n_games = 3000

    fname = 'ACTOR_CRITIC_' + 'cartpole' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        
        while not done:
            #env.render(mode ="Human")
            action = agent.choose_action(observation)
            #print("---------------------- action ----------------------")
            #print(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)

