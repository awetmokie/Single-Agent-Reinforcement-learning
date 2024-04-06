import gym
import highway_env
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')




def check_observation(observation):
    distance = []
    for i in range(1, len(observation)):
        
        if observation[i][0] != 0.0:
            distance.append(math.sqrt((observation[i][1])**2 + (observation[i][2])**2 ) )
    return (np.min(np.array(distance)))


def calculate_ttc(ref_pos, ref_vel):
    """
    Calculates the time-to-collision (TTC) between the reference vehicle (with position ref_pos and velocity ref_vel)
    and the other vehicle (with position other_pos and velocity other_vel).
    """
    # Calculate relative position and velocity
    rel_pos =  ref_pos
    rel_vel =  ref_vel

    # Calculate the time to collision
    dist = np.linalg.norm(rel_pos)
    # print(dist)
    relative_speed = np.dot(rel_vel, abs(rel_pos)) / dist
    ttc = dist / abs(relative_speed) if abs(relative_speed) > 1e-3 else float('inf')
    # ttc = dist / abs(relative_speed) 

    return ttc


def test_collision (observation):
    ttc = []
    for i in range(1, len(observation)):
        if int(observation[i][0]):
            rel_pos = np.array([observation[i][1],observation[i][2]])
            rel_vel = np.array([observation[i][3], observation[i][4]])

            ttc.append(calculate_ttc (rel_pos, rel_vel))
    
    # print(ttc)
    return np.min(np.array(ttc))


if __name__ == '__main__':
    env = gym.make('joker-intersection-v0')

    n_games = 1
    
    for i in range(n_games):
        counter = 0
        obs = env.reset()
        # print(obs)
        score = 0
        done = False
        # print(obs)
        while not done:
            env.render()
            action = env.action_space.sample()
            
            obs_, reward, done, info = env.step(action)
            if int(info['agents_costs'][0]):
                counter += 1
            # print(int(info['agents_costs'][0]))
            
                # print(obs)
                # print(check_observation(obs))
            #print(obs)x
            # print(int(info['agents_costs'][0]))
            d = check_observation(obs)
            print(d)
            # print(test_collision(obs), "action ", action , " cost",int(info['agents_costs'][0]),"reward",reward)
            obs =obs_
            score += reward
            #env.render()
        #print(int(info['agents_costs'][0]))
        print(counter)
        print('episode ', i, 'score %.1f' % score)

