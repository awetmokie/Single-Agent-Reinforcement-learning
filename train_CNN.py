import gym
import highway_env
import math
import matplotlib.pyplot as plt
import numpy as np
from CC_Reinforce_version_3_copy.reinforce_torch_CNN import PolicyGradientAgent as CC_Reinforce_3_PolicyGradientAgent
from CC_Reinforce_version_5.reinforce_torch import PolicyGradientAgent as CC_Reinforce_5_PolicyGradientAgent
from Reinforce.reinforce_torch import PolicyGradientAgent as PolicyGradientAgent
import warnings
import matplotlib
warnings.filterwarnings('ignore')



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
    # env = gym.make('joker-highway-fast-v0')
    # env = gym.make('joker-intersection-v0')
    # env = gym.make('two-way-v0')
    



#     env.configure({
  
#   "vehicles_count": 100,
#   "collision_reward": -1,
#   #"initial_vehicle_count": 3,
#   "duration": 40,  # [s]
#             # "destination": "o1",
#             # "initial_vehicle_count": 5,
#             "ego_spacing": 2,
#             "vehicles_density": 2,
#             # "spawn_probability": 0.6,
#             # "screen_width": 2000,
#             # "screen_height": 600,
#             #"centering_position": [0.5, 0.6],
#             #"scaling": 5.5 * 1.3,
#             #"collision_reward": -2,
#             # "high_speed_reward": 1,
#             # "arrived_reward": 1,
#             # "reward_speed_range": [7.0, 9.0],
#             # "normalize_reward": False,
#             # "offroad_terminal": False,
  
 
# })  
    


    screen_width, screen_height = 84, 84
    config = {
    "offscreen_rendering":True,
    "observation": {
        "type":"GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],
        "stack_size": 4,"observation_shape": (screen_width, screen_height),
    },
    "screen_width": screen_width,
    "screen_height": screen_height,
    "scaling": 1.75,
    "policy_frequency": 2
    }

    env = gym.make('joker-intersection-v0')
    env.configure(config)
    state_size = 2
    # state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
    # print(env.observation_space)
    # print(state_size)
    num_actions = env.action_space.n
    print(num_actions)
    n_games = 2000
    





    print("----------------------------------------------- CC_REINFORCE_VERSION_3_COPY------------------------------------")
    delta = 0.09
    agent = CC_Reinforce_3_PolicyGradientAgent(gamma=0.99, lr=0.00005, input_dims=[state_size],
                                n_actions=num_actions ,delta = delta)
    

    # agent.load_model(  path="CC_Reinforce_version_3_copy/tmp/highway/state_dict_model" +  str(0.01) +".pt")
    

    version_12 = '0.90 + '

    fname = 'cc_REINFORCE_'  + 'single_agent' + str(agent.lr) + '_' \
            + str(n_games) + 'games' 
    figure_file = 'plots/normal/CNN/' + fname + str(delta) + '.png'
    safety_plot_090 = 'plots/normal/CNN/' + 'probability' + str(delta)+ '.png'
    safety_plot = 'plots/normal/CNN/' + 'probability' +'.png'
    fig = 'plots/normal/CNN/' + fname + '.png'
    

    scores = []
    safety_090 = []
    lamda_090 = []

    EVALUATE =  False
    
    if not EVALUATE:

        for i in range(n_games):
            done = False
            observation = env.reset()
            # print(observation)
            score = 0
            while not done:
                env.render()
                # action = agent.choose_action(observation.reshape(state_size))
                action = agent.choose_action(observation)
                # print(action)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                # agent.store_costs(int(info['agents_costs'][0]))
                # cost =  check_observation(observation)
                # cost = cost - cost*int(info['agents_costs'][0])
                # print(info['cost'])
                agent.store_costs(int(info['cost']))
                # agent.store_costs(int(info['agents_costs'][0]))

                # if done:
                # print(check_observation(observation_))

                observation = observation_
            lamda_090.append(agent.learn())
            safety_090.append(int(agent.check_constraints_satisfaction()))
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        label = 'CC_version_3' +' '  + str(delta)
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, fig,label)
        y_090 = [i+1 for i in range(len(safety_090))]

        # label ='lamda 0.90'
        # plot_learning_curve(lamda_090, y_090, safety_plot_090,label)

        # # plt.close()

        # label == 'probability'
        # plot_learning_curve(safety, y, safety_plot,label)

        agent.save_model(  path="CC_Reinforce_version_3_copy/tmp/highway/state_dict_model" +  str(delta) +".pt")   # 12 we have 0.99 trained in 5000
     
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









    


    print("----------------------------------------------- CC_REINFORCE_VERSION_3_COPY------------------------------------")
    delta = 0.01
    agent = CC_Reinforce_3_PolicyGradientAgent(gamma=0.99, lr=0.00005, input_dims=[state_size],
                                n_actions=num_actions ,delta = delta)
    

    version_12 = '0.90 + '

    fname = 'cc_REINFORCE_' + 'single_agent' + str(agent.lr) + '_' \
            + str(n_games) + 'games' 
    figure_file = 'plots/normal/CNN/' + fname +  str(delta) + '.png'
    safety_plot_099 = 'plots/normal/CNN/' + 'probability' + str(delta)+ '.png'

    scores = []
    safety_099 = []

    EVALUATE =  False
    lamda_099 = []
    
    if not EVALUATE:

        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                # env.render()
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                # agent.store_costs(int(info['agents_costs'][0]))
                # cost =  check_observation(observation)
                # cost = cost - cost*int(info['agents_costs'][0])
                # agent.store_costs(int(info['agents_costs'][0]))
                # print(info)
                agent.store_costs(int(info['cost']))

                # if done:
                # print(check_observation(observation_))

                observation = observation_
            lamda_099.append(agent.learn())
            safety_099.append(int(agent.check_constraints_satisfaction()))
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        label = 'CC_version_3' + ' ' + str(delta)
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, fig,label)
        y_099 = [i+1 for i in range(len(safety_099))]
        # # plt.close()

        # label == 'probability'
        # plot_learning_curve(safety, y, safety_plot,label)

        agent.save_model(  path="CC_Reinforce_version_3_copy/tmp/highway/state_dict_model" +  str(delta) +".pt")   # 12 we have 0.99 trained in 5000
     
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




    







    print("----------------------------------------------- CC_REINFORCE_VERSION_3_COPY------------------------------------")
    delta = 0.05
    agent = CC_Reinforce_3_PolicyGradientAgent(gamma=0.99, lr=0.00005, input_dims=[state_size],
                                n_actions=num_actions ,delta = delta)
    

    version_12 = '0.90 + '

    fname = 'cc_REINFORCE_' + version_12 + 'single_agent' + str(agent.lr) + '_' \
            + str(n_games) + 'games' 
    figure_file = 'plots/normal/CNN/' + fname + str(delta) +  '.png'
    safety_plot_095 = 'plots/normal/CNN/' + 'probability' + str(delta)+ '.png'

    scores = []

    safety_095 = []
    lamda_095 = []

    EVALUATE =  False
    
    if not EVALUATE:

        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                # env.render()
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                # agent.store_costs(int(info['agents_costs'][0]))
                # cost =  check_observation(observation)
                # cost = cost - cost*int(info['agents_costs'][0])
                # agent.store_costs(int(info['agents_costs'][0]))
                # print(info)
                agent.store_costs(int(info['cost']))

                # if done:
                # print(check_observation(observation_))

                observation = observation_
            lamda_095.append(agent.learn())
            safety_095.append(int(agent.check_constraints_satisfaction()))
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score)
        label = 'CC_version_3' + ' ' + str(delta)
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, fig,label)
        y_095 = [i+1 for i in range(len(safety_095))]


        plt.close()

        label ='CC_probability 0.10'
        plot_learning_curve(safety_090, y_090, safety_plot,label)

        label ='CC_probability 0.99'
        plot_learning_curve(safety_099, y_099, safety_plot,label)

        plt.axhline(y= 0.1, color='red', linestyle='dotted')

        plt.axhline(y= 0.01, color='green', linestyle='dotted')


        label ='CC_probability 0.95'
        plot_learning_curve(safety_095, y_095, safety_plot,label)

        plt.close()

        lamda_plot = 'plots/normal/CNN/' + 'lamda' + '.png'

        label ='lamda 0.90'
        plot_learning_curve(lamda_090, y_090, lamda_plot,label)


        label ='lamda 0.99'
        plot_learning_curve(lamda_099, y_099, lamda_plot,label)

        label ='lamda 0.95'
        plot_learning_curve(lamda_095, y_095, lamda_plot,label)



        # label == 'probability'
        # plot_learning_curve(safety, y, safety_plot,label)

        agent.save_model(  path="CC_Reinforce_version_3_copy/tmp/highway/state_dict_model" +  str(delta) +".pt")   # 12 we have 0.99 trained in 5000
     
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
                print(info)
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













        
     




    # print("=======================================REINFORCE=====================================================")



    # agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[state_size],
    #                             n_actions=5)

    # fname = 'REINFORCE_' + 'single_agent' + str(agent.lr) + '_' \
    #         + str(n_games) + 'games'
    # figure_file = 'plots/' + fname + '.png'
    # safety_plot = 'plots/' + 'probability' + '.png'

    # scores = []
    # reinforce_safety = []
    # for i in range(n_games):
    #     done = False
    #     observation = env.reset()
    #     score = 0
    #     while not done:
    #         # env.render()
    #         action = agent.choose_action(observation.reshape(state_size))
    #         observation_, reward, done, info = env.step(action)
    #         score += reward
    #         agent.store_rewards(reward)
    #         agent.store_costs(int(info['agents_costs'][0]))

    #         observation = observation_
    #     agent.learn()
    #     reinforce_safety.append(int(agent.check_constraints_satisfaction()))
    #     scores.append(score)

    #     avg_score = np.mean(scores[-100:])
    #     print('episode ', i, 'score %.2f' % score,
    #             'average score %.2f' % avg_score)

    # x = [i+1 for i in range(len(scores))]
    # label = 'REINFORCE'
    # plot_learning_curve(scores, x, figure_file,label)
    # agent.save_model(  path="Reinforce/tmp/highway/state_dict_model.pt")




    # print(" --------------------------------CC_REINFORCE_VERSION_5 ----------------------------------------------------")


    # agent = CC_Reinforce_5_PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[state_size],
    #                             n_actions= num_actions)

    # fname = 'CC_REINFORCE_VERSION_5' + 'single_agent' +'version_1'+ str(agent.lr) + '_' \
    #         + str(n_games) + 'games'
    # figure_file = 'plots/' + fname + '.png'

    # risk_fig = 'plots/' + 'risk' + '.png'

    # scores = []
    # estimated_risk = []

    # EVALUATE = False
    # # agent.load_model(path ="state_dict_model_intersection_v0.pt")
    # if not EVALUATE:
    #     for i in range(n_games):
    #         done = False
    #         observation = env.reset()
    #         score = 0
    #         samples =[]
    #         while not done:
    #             # env.render()
    #             action,_ = agent.choose_action(observation.reshape(state_size))
    #             observation_, reward, done, info = env.step(action)
    #             score += reward
    #             agent.store_rewards(reward)
    #             # agent.store_costs(float(info['agents_costs'][0]))
    #             # cost = check_observation(observation)
    #             cost = int(info['agents_costs'][0])
    #             agent.store_costs(cost)
    #             # print (cost)
                
    #             samples.append((observation.reshape(state_size), _, reward,cost, observation_.reshape(state_size),done))

    #             if done:
    #                 pass
    #                 # print(check_observation(observation_))

    #             observation = observation_
    #         estimated_risk.append(agent.updated_learn(samples))
    #         scores.append(score)

    #         avg_score = np.mean(scores[-100:])
    #         print('episode ', i, 'score %.2f' % score,
    #                 'average score %.2f' % avg_score)
        
    #     x = [i+1 for i in range(len(scores))]
    #     label = "CC_Version_5"
    #     plot_learning_curve(scores, x, figure_file,label)
    #     x = [i+1 for i in range(len(estimated_risk))]



    #     plt.close()

    #     label ='CC_probability 0.90'
    #     plot_learning_curve(safety_090, y_090, safety_plot_090,label)

    #     label ='CC_probability 0.99'
    #     plot_learning_curve(safety_099, y_099, safety_plot_099,label)

    #     label ='CC_probability 0.95'
    #     plot_learning_curve(safety_095, y_095, safety_plot_095,label)



    #     y = [i+1 for i in range(len(reinforce_safety))]

    #     label = 'Reinforce_Probability'

    #     plot_learning_curve(reinforce_safety, y, safety_plot,label)



    #     label  = "Estimatedrisk"
    #     plot_learning_curve(estimated_risk, x, risk_fig,label)
    #     agent.save_model(  path="CC_Reinforce_version_5/tmp/highway/state_dict_model.pt") # version 0 was info cost 
    
    # else:
    #     # agent.load_model(path ="state_dict_model_intersection_v0.pt")
    #     scores = []
    #     eval_games = 100
    #     counter = 0 
    #     time_step = 0
    #     sum =  []
    #     for i in range(eval_games):
    #         done = False
    #         observation = env.reset()
    #         score = 0
    #         time_step = 0
    #         # counter = 0
    #         while not done:
    #             time_step += 1
    #             # env.render()
    #             action = agent.choose_action(observation.reshape(state_size))
    #             observation_, reward, done, info = env.step(action)
    #             score += reward
    #             # agent.store_rewards(reward)
    #             # cost = check_observation(observation) - 2* int(info['agents_costs'][0])
    #             # if (cost >= 0.0):
    #             #     counter +=1
    #                 # break
    #             agent.store_costs(float(info['agents_costs'][0]))
    #             observation = observation_

    #         check = agent.check_constraints_satisfaction()
    #         sum.append(check)

    #         # agent.learn()
    #         if check:
    #             counter += 1
    #         scores.append(score)

    #         avg_score = np.mean(scores[-100:])
    #         print('episode ', i, 'score %.2f' % score,
    #                 'average score %.2f' % avg_score)
        

    #     print('the number of constraint violation is :',counter)

    #     print('the probability of constraint SATISFACTICTION  on average is :',(sum)) 
    #     # print('the number of constraint violations in this run is :',counter )

        
    

    

    




        
    

    

    