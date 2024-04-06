import gym
import highway_env
import math
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
    plt.axhline(y=0.99, color='red', linestyle='dotted')
    plt.axhline(y=0.1, color='green', linestyle='dotted')
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def check_observation(observation):
    distance = []
    for i in range(1, len(observation)):

        distance.append(math.sqrt((observation[i][1])**2 ))
    return int(np.min(np.array(distance))<= 0.07)

if __name__ == '__main__':
    env = gym.make('joker-highway-v0')

    state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
    print(state_size)
    num_actions = env.action_space.n
    n_games = 1000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[state_size],
                                n_actions= num_actions)

    fname = 'REINFORCE_' + 'single_agent' +'version_1'+ str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    risk_fig = 'plots/' + 'risk' + '.png'

    scores = []
    estimated_risk = []

    EVALUATE = False
    # agent.load_model(path ="state_dict_model_intersection_v0.pt")
    if not EVALUATE:
        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            samples =[]
            while not done:
                # env.render()
                action,_ = agent.choose_action(observation.reshape(state_size))
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                # agent.store_costs(float(info['agents_costs'][0]))
                cost = check_observation(observation)
                agent.store_costs(cost)
                # print (cost)
                
                samples.append((observation.reshape(state_size), _, reward,cost, observation_.reshape(state_size),done))

                if done:
                    pass
                    # print(check_observation(observation_))

                observation = observation_
            estimated_risk.append(agent.updated_learn(samples))
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, figure_file)
        x = [i+1 for i in range(len(estimated_risk))]
        plot_learning_curve(estimated_risk, x, risk_fig)
        agent.save_model(  path="state_dict_model_intersection_v1.pt") # version 0 was info cost 
    
    else:
        agent.load_model(path ="state_dict_model_intersection_v0.pt")
        scores = []
        eval_games = 100
        counter = 0 
        time_step = 0
        sum =  []
        for i in range(eval_games):
            done = False
            observation = env.reset()
            score = 0
            time_step = 0
            # counter = 0
            while not done:
                time_step += 1
                # env.render()
                action = agent.choose_action(observation.reshape(state_size))
                observation_, reward, done, info = env.step(action)
                score += reward
                # agent.store_rewards(reward)
                # cost = check_observation(observation) - 2* int(info['agents_costs'][0])
                # if (cost >= 0.0):
                #     counter +=1
                    # break
                agent.store_costs(float(info['agents_costs'][0]))
                observation = observation_

            check = agent.check_constraints_satisfaction()
            sum.append(check)

            # agent.learn()
            if check:
                counter += 1
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        

        print('the number of constraint violation is :',counter)

        print('the probability of constraint SATISFACTICTION  on average is :',(sum)) 
        # print('the number of constraint violations in this run is :',counter )

        
    

    

    