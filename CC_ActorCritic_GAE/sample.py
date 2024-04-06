import gym
import torch
import highway_env
import numpy as np
from actor_critic_torch import Agent
from utils import plot_learning_curve
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = gym.make('highway-v0')
    env.configure({
    "action": {
        "type": "DiscreteMetaAction"
       }
    })

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

    fname = 'ACTOR_CRITIC_' + 'lunar_lander_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        samples = []
        while not done:
            env.render(mode ="Human")
            action ,_ = agent.choose_action(observation.reshape(input_dims))
            #print("---------------------- action ----------------------")
            #print(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            # Collect samples
            samples.append((observation.reshape(input_dims), _, reward, observation_.reshape(input_dims)))

            # agent.learn(observation.reshape(input_dims), reward, observation_.reshape(input_dims), done)
            observation = observation_
        scores.append(score)
        
        # # Transpose our samples
        # states, actions, rewards, next_states = zip(*samples)

        # states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
        # next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
        # actions = torch.as_tensor(actions).unsqueeze(1)
        # rewards = torch.as_tensor(rewards).unsqueeze(1)


        agent.updated_learn(samples)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)

