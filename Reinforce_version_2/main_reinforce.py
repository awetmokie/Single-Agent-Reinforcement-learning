import gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from reinforce_torch import PolicyGradientAgent
import warnings
warnings.filterwarnings('ignore')

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('joker-highway-v0')

    state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
    print(state_size)
    num_actions = env.action_space.n
    n_games = 1000
    num_trajectories = 10
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[state_size],
                                n_actions=5 , num_trajectories = num_trajectories)

    fname = 'REINFORCE_version_2' + 'single_agent' + str(agent.lr) + '_' \
            + str(n_games) + 'games' + str(num_trajectories) + 'trajectories'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        cummulative_score = []
        for j in range(num_trajectories):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                # env.render()
                action = agent.choose_action(observation.reshape(state_size))
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                observation = observation_
            agent.calulate_loss()
            cummulative_score.append(score)
        agent.learn()
        scores.append(np.mean(cummulative_score))

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
    agent.save_model(  path="state_dict_model.pt")
