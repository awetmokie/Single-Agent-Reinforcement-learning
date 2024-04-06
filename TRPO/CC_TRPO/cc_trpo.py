import gym
import highway_env
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
import matplotlib
import warnings
warnings.filterwarnings('ignore')




class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        #self.v = nn.Linear(fc2_dims, 1)
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        #v = self.v(x) 

        return (pi)
    
    def save_model(self,policy, path= "state_dict_model.pt"):
        # Specify a path
        self.PATH = path

        # Save
        T.save(policy.state_dict(), self.PATH)
    
    def load_model(self ,path="state_dict_model.pt"):
        if path :
            self.PATH = path  
        self.load_state_dict(T.load(self.PATH))

    
    


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, n_agents=2,fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        #self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #pi = self.pi(x)
        v = self.v(x)

        return (v)




max_d_kl = 0.01

class Agent():
    def __init__(self,actor_dims,critic_dims , n_actions, n_agents, agent_idx, alpha = 5e-4, beta = 5e-4 ,
                 gamma=0.99,fc1_dims= 256, fc2_dims=256,chkpt_dir='tmp/trpo/'):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.agent_idx = agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.critic = CriticNetwork(beta, critic_dims, n_actions,n_agents= n_agents, 
                                               fc1_dims = fc1_dims, fc2_dims = fc2_dims)
        self.log_prob = None
        self.lamda =  T.tensor(100.0)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        probabilities = self.actor.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        #print(state.size())
        #print(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()
    
    def save_model(self,path= "state_dict_model.pt"):
        # Specify a path
        self.actor.save_model(self.actor,path)
    
    def load_model(self ,path="state_dict_model.pt"):

        self.actor.load_model(path)

    def learn(self, state,reward, state_ ,done):
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()
        
        critic_loss = delta**2

        (critic_loss).backward(retain_graph = True)
        self.critic.optimizer.step()

    def update_critic(self,advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()
    

    def update_agent(self,rollouts):
        states = T.cat([r.states for r in rollouts], dim=0)
        actions = T.cat([r.actions for r in rollouts], dim=0).flatten()
        
        advantages = [self.estimate_advantages(states, next_states[-1], rewards,costs) for states, _, rewards, next_states,costs in rollouts]
        advantages = T.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.update_critic(advantages)

        count = 0
        probs = [self.estimate_probs(states, next_states[-1], costs) for states, _, rewards, next_states,costs in rollouts]
        for i in probs:
            if i >= 1:
                count += 1
        probs = count/len(probs)
        print('probs:', probs)
        print('lamda:', self.lamda)

        self.lamda += 0.5*(probs -0.01 )


        
        distribution = self.actor.forward(states)
        distribution = F.softmax(distribution, dim=1)
        
        distribution = T.distributions.utils.clamp_probs(distribution)
        #probabilities = T.distributions.Categorical(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = self.kl_div(distribution, distribution)

        parameters = list(self.actor.parameters())

        g = self.flat_grad(L, parameters, retain_graph=True)
        d_kl = self.flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return self.flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = T.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.apply_update(step)

            with T.no_grad():
                distribution_new = self.actor.forward(states)
                distribution_new = F.softmax(distribution_new, dim=1)
                distribution_new = T.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                L_new = self.surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= max_d_kl:
                return True

            self.apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1


    def estimate_advantages(self,states, last_state, rewards,costs):
        values = self.critic(states)
        last_value = self.critic(last_state.unsqueeze(0))
        next_values = T.zeros_like(rewards)
        
        for i in reversed(range(rewards.shape[0])):
            if costs[i] >= 1.0:
                 last_value = next_values[i] = (rewards[i]- self.lamda) + 0.99 * last_value
            else:
                last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        return advantages

    def estimate_probs(self,states, last_state, costs):
        next_cost = T.zeros_like(costs)
        total_cost = T.tensor(0.0)
        for i in reversed(range(costs.shape[0])):
            # print(costs[i])
            next_cost[i] =costs[i]
            total_cost += costs[i].item()
            pass
        # print(total_cost)
        return total_cost
        return next_cost
    




      

    def surrogate_loss(self,new_probabilities, old_probabilities, advantages):
        return (new_probabilities / old_probabilities * advantages).mean()


    def kl_div(self,p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()


    def flat_grad(self,y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = T.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = T.cat([t.view(-1) for t in g])
        return g


    def conjugate_gradient(self,A, b, delta=0., max_iterations=10):
        x = T.zeros_like(b)
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


    def apply_update(self,grad_flattened):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            #print( p.data)
            #print("---------g---------")
            #print(g)
            p.data += g
            #print( p.data)
            
            n += numel


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







if __name__ == '__main__':
    env = gym.make('two-way-v0')
    state_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    num_actions = env.action_space.n

    actor_dims = critic_dims = state_size
    n_agents = 1


    Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states','costs' ])
    epochs=5
    num_rollouts=20
    agent = Agent(actor_dims, critic_dims,  
                            num_actions, n_agents, agent_idx=0,alpha=5e-4, beta=5e-4,
                            gamma =0.99 , fc1_dims=256, fc2_dims=256,chkpt_dir='tmp/trpo/')



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

                # env.render()

                with T.no_grad():
                    action = agent.choose_action(state.reshape(state_size))

                next_state, reward, done, _ = env.step(action)

                # Collect samples
                samples.append((state.reshape(state_size), action, reward, next_state.reshape(state_size),_['cost']))

                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states,costs = zip(*samples)

            states = T.stack([T.from_numpy(state) for state in states], dim=0).float()
            next_states = T.stack([T.from_numpy(state) for state in next_states], dim=0).float()
            actions = T.as_tensor(actions).unsqueeze(1)
            rewards = T.as_tensor(rewards).unsqueeze(1)
            costs = T.as_tensor(costs).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states,costs))

            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        agent.update_agent(rollouts)
        mtr = np.mean(rollout_total_rewards)
        print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')

        mean_total_rewards.append(mtr)
    
    figure_file = 'plot/CC_TRPO.png'
    label = 'CC_TRPO'
    x = [i+1 for i in range(len(mean_total_rewards))]
    plot_learning_curve(mean_total_rewards, x, figure_file,label)


    agent.save_model(path= "tmp/2-way/state_dict_model_trpo.pt")



