import gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# env = gym.make('CartPole-v0')

# env = gym.make('highway-v0')
env = gym.make('two-way-v0')
# env = gym.make('joker-intersection-v0')


# state_size = env.observation_space.shape[0]

state_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
num_actions = env.action_space.n



Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', 'costs'])




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


def train(epochs=500, num_rollouts=20, render_frequency=None):
    mean_total_rewards = []
    global_rollout = 0

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        for t in range(num_rollouts):
            state = env.reset()
            done = False

            samples = []

            while not done:
                # if render_frequency is not None and global_rollout % render_frequency == 0:
                #     env.render()

                env.render()

                with torch.no_grad():
                    action = get_action(state.reshape(state_size))

                next_state, reward, done, _ = env.step(action)

                # Collect samples
                samples.append((state.reshape(state_size), action, reward, next_state.reshape(state_size),_['cost']))

                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states,costs = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)
            costs = torch.as_tensor(costs).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states,costs))

            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        update_agent(rollouts)
        mtr = np.mean(rollout_total_rewards)
        print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')

        mean_total_rewards.append(mtr)
    figure_file = 'cc_trpo_intersection.png'
    label = 'cc_trpo'
    x = [i+1 for i in range(len(mean_total_rewards))]
    plot_learning_curve(mean_total_rewards, x, figure_file,label)

    # plt.plot(mean_total_rewards)
    # plt.savefig('cc_trpo.png')
    # # plt.show()


actor_hidden =  64
actor = nn.Sequential(nn.Linear(state_size, actor_hidden),
                      nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))


def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item()


# Critic takes a state and returns its values
critic_hidden = 64
critic = nn.Sequential(nn.Linear(state_size, critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))
critic_optimizer = Adam(critic.parameters(), lr=0.0005)


def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()


# delta, maximum KL divergence
max_d_kl = 0.01
global lamda 
lamda = 0


def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards,costs) for states, _, rewards, next_states,costs in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()
    


    update_critic(advantages)


    # cost function 

    cost_value = [estimate_costs(states, next_states[-1], costs) for states, _, rewards, next_states,costs in rollouts]
    cost_value = torch.cat(cost_value , dim=0).flatten()
    count = 0
    # print(cost_value)
    # for i in cost_value:
    #     print(i)
    #     if i >= 1 :

    #         count += 1
    
    # print(count,len(cost_value))
    # probs = count/len(cost_value)
    # print(probs)

    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]
    # print(probabilities.shape)

    # Now we have all the data we need for the algorithm

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant

    global lamda
    # cost_value = (cost_value - cost_value.mean()) / cost_value.std()

    probs = [estimate_probs(states, next_states[-1], costs) for states, _, rewards, next_states,costs in rollouts]
    for i in probs:
        if i >= 1:
            count += 1
    probs = count/len(probs)
    print('probs:', probs)
    print('lamda:', lamda)

    # if probs > 0.05 :
    # advantages = advantages - lamda*cost_value

    lamda += 0.5*(probs -0.01 )
    
    
    
    # if probs > 0.01:
    #     advantages = advantages 
        
    #     lamda += 0.5*(probs-0.01)
        
    # else:
    #     advantages = advantages 
    #     lamda -= 0.5*(0.3)

    # if lamda < 0:
    #     lamda = 0
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    # print(L.shape)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1


def estimate_advantages(states, last_state, rewards,costs):
    values = critic(states)
    global lamda
    # print(rewards.shape)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        if costs[i] >= 1.0:
            last_value = next_values[i] = (rewards[i]- lamda) + 0.99 * last_value
        else:
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
    advantages = next_values - values
    return advantages

def estimate_probs(states, last_state, costs):
    
    # print(costs[0])
    next_cost = torch.zeros_like(costs)
    total_cost =  torch.tensor(0.0)
    for i in reversed(range(costs.shape[0])):
        # print(costs[i])
        next_cost[i] =costs[i]
        total_cost += costs[i].item()
        pass
    # print(total_cost)
    return total_cost
    return next_cost

def estimate_costs(states, last_state, costs):
    
    # print(costs[0])
    next_cost = torch.zeros_like(costs)
    total_cost =  torch.tensor(0.0)
    for i in reversed(range(costs.shape[0])):
        # print(costs[i])
        next_cost[i] =costs[i]
        total_cost += costs[i].item()
        pass
    # print(total_cost)
    if total_cost >= 1.0:
        return   torch.ones_like(costs)
    
    return torch.zeros_like(costs)

    return total_cost
    return next_cost


def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel


def save_model(policy, path= "state_dict_model.pt"):
        # Specify a path
        PATH = path

        # Save
        torch.save(policy.state_dict(), PATH)

def load_model(path="state_dict_model.pt"):
        print(path)
        if path :
            PATH = path  
        actor.load_state_dict(torch.load(PATH))


def test_action(state):
    
    state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item()


def test(epochs=500, num_rollouts=20, render_frequency=None):
    scores =[]
    counter = 0
    load_model(path="state_dict_model_v3.pt")
   
    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []
        score =0
        done = False
        state = env.reset()
        while not done:
               

            env.render()

            with torch.no_grad():
                action = test_action(state.reshape(state_size))

            next_state, reward, done, _ = env.step(action)
            if float(_['cost'])>= 1 :
                counter += 1

            state = next_state

            score += reward
              

               
           
           
            
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', epoch, 'score %.2f' % score,
                    'average score %.2f' % avg_score)

    print('total collision ' ,counter)

        


# Train our agent
train(epochs=500, num_rollouts=20, render_frequency=50)
# save_model(actor,path= "state_dict_model_intersection_v3.pt")    # v2 model is 20 num_rollouts  v3 model is 50 num_rollouts

# Evaluate our agent 
# test(epochs=500, num_rollouts=10, render_frequency=50)



