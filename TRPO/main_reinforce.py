import gym
import highway_env
import math
import matplotlib.pyplot as plt
import numpy as np
from reinforce_torch import PolicyGradientAgent
import warnings
import matplotlib
warnings.filterwarnings('ignore')

# def plot_learning_curve(scores, x, figure_file):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.axhline(y=0.99, color='red', linestyle='dotted')
#     plt.axhline(y=0.90, color='green', linestyle='dotted')
#     plt.title('Running average of previous 100 scores')
#     plt.savefig(figure_file)
#     plt.close()


def plot_learning_curve(scores,x, figure_file ,label):
    running_avg = np.zeros(len(scores))
    matplotlib.style.use('seaborn-v0_8')
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg,label=label )

    if label == 'probability':
         plt.axhline(y=0.99, color='red', linestyle='dotted')
         plt.axhline(y=0.90, color='green', linestyle='dotted')

   
    plt.title('Running average of previous 100 scores')
    plt.legend()
    plt.savefig(figure_file)
    # plt.close()


def check_observation(observation):
    distance = []
    for i in range(1, len(observation)):
        

        distance.append(math.sqrt((observation[i][1])**2 ))
    return np.min(np.array(distance))

if __name__ == '__main__':
    env = gym.make('joker-highway-v0')

    state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
    print(state_size)
    num_actions = env.action_space.n
    n_games = 2000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[state_size],
                                n_actions=5)
    

    version_12 = '0.90 + '

    fname = 'cc_REINFORCE_' + version_12 + 'single_agent' + str(agent.lr) + '_' \
            + str(n_games) + 'games' 
    figure_file = 'plots/' + fname + 'night.png'
    safety_plot = 'plots/' + 'probability_night' +'.png'

    scores = []
    safety = []

    EVALUATE =  False
    
    if not EVALUATE:




        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                # env.render()
                action = agent.choose_action(observation.reshape(state_size))
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                # agent.store_costs(int(info['agents_costs'][0]))
                cost =  check_observation(observation)
                # cost = cost - cost*int(info['agents_costs'][0])
                agent.store_costs(int(info['agents_costs'][0]))

                # if done:
                # print(check_observation(observation_))

                observation = observation_
            agent.learn()
            safety.append(int(agent.check_constraints_satisfaction()))
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        label = 'CC_Reinforce'
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, figure_file,label)
        y = [i+1 for i in range(len(safety))]
        plt.close()

        label == 'probability'
        plot_learning_curve(safety, y, safety_plot,label)

        agent.save_model(  path="state_dict_model_v13.pt")   # 12 we have 0.99 trained in 5000
     
    else:
        agent.load_model(path ="state_dict_model_v11.pt")
        scores = []
        eval_games = 1000
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
                agent.store_costs(int(info['agents_costs'][0]))
                
                # cost =  check_observation(observation)
                # cost = cost - cost*int(info['agents_costs'][0])
                
                # if (cost >= 0.01):
                #     counter +=1
                
                observation = observation_
            

            if not agent.check_constraints_satisfaction():
                counter += 1

            
            # sum.append(counter/(time_step))
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,'time step',time_step)
        

        print('the probability of constraint SATISFACTICTION is :',1 - (counter/eval_games))

        print('the probability of constraint violation  :', counter/eval_games) 
        print('the number of  collisions in %d',eval_games,'is ' ,counter)
        # print('the number of constraint violations in this run is :',counter )

        
    

    

    