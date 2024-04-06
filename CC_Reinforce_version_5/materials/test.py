import gym
import highway_env
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')




def check_observation(observation):
    distance = []
    for i in range(1, len(observation)):

        distance.append(math.sqrt((observation[i][1])**2 ))
    return np.min(np.array(distance))


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
    relative_speed = np.dot(rel_vel, rel_pos) / dist
    ttc = dist / abs(relative_speed) 

    return ttc


def test_collision (observation):
    ttc = []
    for i in range(1, len(observation)):

        rel_pos = np.array([observation[i][1],observation[i][2]])
        rel_vel = np.array([observation[i][3], observation[i][4]])

        ttc.append(calculate_ttc (rel_pos, rel_vel))
    
    # print(ttc)
    return np.min(np.array(ttc))


if __name__ == '__main__':
    env = gym.make('joker-highway-v0')

    n_games = 1

    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            
            obs_, reward, done, info = env.step(action)
            # print(obs_)
            print(check_observation(obs))
            # print(test_collision(obs), "action ", action , " cost",int(info['agents_costs'][0]))
            obs =obs_
            score += reward
            #env.render()
        #print(int(info['agents_costs'][0]))
        print('episode ', i, 'score %.1f' % score)

