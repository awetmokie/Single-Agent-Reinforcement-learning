import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)
    
    def save_model(self,policy, path= "state_dict_model.pt"):
        # Specify a path
        self.PATH = path

        # Save
        T.save(policy.state_dict(), self.PATH)
    
    def load_model(self ,path="state_dict_model.pt"):
        if path :
            self.PATH = path  
        self.load_state_dict(T.load(self.PATH))
    


class CostCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(CostCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        v = self.v(x)

        return  v
    
    def save_model(self,policy, path= "state_dict_model.pt"):
        # Specify a path
        self.PATH = path

        # Save
        T.save(policy.state_dict(), self.PATH)
    
    def load_model(self ,path="state_dict_model.pt"):
        if path :
            self.PATH = path  
        self.load_state_dict(T.load(self.PATH))
    

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
                 gamma=0.99,threshold =0.03 , delta = 0.1):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        
        self.cost_critic = CostCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None
        self.lamda =  T.tensor(0.0, requires_grad=True).to(self.cost_critic.device)
        self.threshold = threshold
        self.delta = delta 

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item(),log_prob
    


    def save_model(self,path= "state_dict_model.pt"):
        # Specify a path
        self.actor_critic.save_model(self.actor_critic,path)
    
    def load_model(self ,path="state_dict_model.pt"):

        self.actor_critic.load_model(path)

    def learn(self, state, reward, state_, done,cost):
        self.actor_critic.optimizer.zero_grad()
        self.cost_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        cost = T.tensor(cost, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        cost_critic_value = self.cost_critic.forward(state)
        cost_critic_value_ = self.cost_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        delta_cost = cost + cost_critic_value_*(1-int(done)) - cost_critic_value

        actor_loss = -(self.log_prob*delta - self.log_prob*self.lamda *delta_cost)
        critic_loss = delta**2
        

        (actor_loss + critic_loss).backward(retain_graph = True)
        self.actor_critic.optimizer.step()



        cost_loss = delta_cost**2
        (cost_loss).backward(retain_graph = True)
        self.cost_critic.optimizer.step()

        
        self.lamda = self.lamda + 0.02*(cost - self.delta)

        self.lamda  = T.max(self.lamda.data, T.tensor(0.0).to(self.cost_critic.device))
        print("self.lamda: ", self.lamda)
    

    def estimate_advantages(self , state , state_ ,rewards):
        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
    


    def compute_returns(self,next_value, rewards):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * 1
            returns.insert(0, R)
        return returns
    
    def updated_learn(self ,samples):
        

        self.actor_critic.optimizer.zero_grad()
        self.cost_critic.optimizer.zero_grad()
        # Transpose our samples
        #states, actions, rewards, next_states = zip(*samples)
        
        states_values  = []
        next_state_values = []
        rewards  = []
        log_probs = []
        advantages =[]
        last_state = None
        for state ,action , reward , next_state in samples:
            state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
            next_state = T.tensor([next_state], dtype=T.float).to(self.actor_critic.device)
            _, critic_value = self.actor_critic.forward(state)
            _, critic_value_ = self.actor_critic.forward(next_state)
            delta = reward + self.gamma*critic_value_*(1) - critic_value
            advantages.append(delta)
            last_state = critic_value_
            rewards.append(T.tensor([reward], dtype=T.float).to(self.actor_critic.device))
            log_probs.append(action)
            states_values.append(critic_value)
            next_state_values.append(critic_value)
        # print("---------- advantages -----------")
        

        returns =self.compute_returns(last_state , rewards)
        returns = T.cat(returns).detach()
        states_values = T.cat(states_values)
        #print("---------- size of the tensors  -----------")
        #advantages = T.cat(advantages, dim=0).flatten()
        advantages = returns - states_values
        #advantages = T.cat(advantages, dim=0).flatten()
        #print(advantages.size())
        log_probs = T.cat(log_probs)
        #print(log_probs.size())
        #advantages = (advantages - advantages).mean() / advantages.std()
        critic_loss =  .5 * (advantages ** 2).mean()
        
        # print(advantages)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()





        policy_loss = []
    
        #calculate loss to be backpropagated

        # for d, lp in zip(advantages, log_probs):
        # #add negative sign since we are performing gradient ascent

        #     policy_loss.append(-d * lp)
        #     c_loss = d**2
        #     a_loss = -d *lp
        #     (c_loss + a_loss).backward(retain_graph=True)
            
        # for d ,lp in zip (advantages ,log_probs):
        #     self.actor_critic.optimizer.step()

    
        #actor_loss = sum(policy_loss)
        
        # (actor_loss + critic_loss).backward()
        # self.actor_critic.optimizer.step()


            

            


        

















