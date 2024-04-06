import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
    def save_model(self,policy, path= "state_dict_model.pt"):
        # Specify a path
        self.PATH = path

        # Save
        T.save(policy.state_dict(), self.PATH)
    
    def load_model(self ,path="state_dict_model.pt"):
        if path :
            self.PATH = path  
        self.load_state_dict(T.load(self.PATH))

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4 ,lamda = 2 ,threshold = 0.1 ,delta = 0.1) :
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        self.cost_memory = []
        self.lamda =  T.tensor(0.0)
        self.threshold = threshold
        self.entropy = 0
        self.delta =delta
        self.entropy_coef = 0.9
        self.entropy_decay = 0.99

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        probabilities = T.distributions.utils.clamp_probs(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.entropy += action_probs.entropy()
        self.action_memory.append(log_probs)
        # print(log_probs)

        return action.item() 

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def store_costs(self, cost):
        self.cost_memory.append(cost)
    

    def save_model(self,path= "state_dict_model.pt"):
        # Specify a path
        self.policy.save_model(self.policy,path)
    
    def load_model(self ,path="state_dict_model.pt"):

        self.policy.load_model(path)
    

    def check_constraints_satisfaction(self):
        C = np.zeros_like(self.cost_memory, dtype=np.float64)
        #C_sum = 0
        for t in range(len(self.cost_memory)):
            C_sum = 0
            discount = 1
            for k in range(t, len(self.cost_memory)):
                C_sum += self.cost_memory[k]*discount
                # discount *= self.gamma
            C[t] = C_sum
            break 
            
        # print(C[0])
        # print("================================================")
        C = T.tensor(C, dtype=T.float).to(self.policy.device)
        # print(C[0])
        prob = float(C[0]) 
        self.cost_memory = []

        return prob

        if (C[0]>= self.threshold):
            self.cost_memory = []
            # print(False)

            return False
        self.cost_memory = []
        return True
    

    def estimate_advantages(self,states, last_state, rewards,costs):
        # values = self.critic(states)
        # last_value = self.critic(last_state.unsqueeze(0))
        next_values = T.zeros_like(rewards)
        discount = self.gamma
        last_value = 0
        # print(costs[-1])

        
        for i in reversed(range(rewards.shape[0])):
            # print(i)
            if costs[i] >= 1.0:
                 last_value = next_values[i] = (rewards[i]- self.lamda) + last_value*discount
                 
            else:
                last_value = next_values[i] = rewards[i] + last_value*discount
            
            # discount *= self.gamma


        # for t in range(rewards.shape[0]):
        #     G_sum = 0
        #     discount = 1
        #     for k in range(t, rewards.shape[0]):
        #         if costs[t] >= 1.0:
        #             G_sum += (rewards[k] - self.lamda)* discount 
        #         else:
        #             G_sum += (rewards[k]* discount) 
        #         discount *= self.gamma
        #     next_values[t] = G_sum
        
        # print(next_values)
        # print ("=======")
        advantages = next_values
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
        
    

    def learn(self,rollouts):
        self.policy.optimizer.zero_grad()

        states = T.cat([r.states for r in rollouts], dim=0)
        actions = T.cat([r.actions for r in rollouts], dim=0).flatten()
        # probs = [self.estimate_probs(states, next_states[-1], costs) for states, _, rewards, next_states,costs in rollouts]

        advantages = [self.estimate_advantages(states, next_states[-1], rewards,costs) for states, _, rewards, next_states,costs in rollouts]
        # print(advantages)
        advantages = T.cat(advantages, dim=0).flatten()
        # print(advantages)

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean())/ advantages.std()

        count = 0
        probs = [self.estimate_probs(states, next_states[-1], costs) for states, _, rewards, next_states,costs in rollouts]
        for i in probs:
            if i >= 1:
                count += 1
        probs = count/len(probs)
        print('probs:', probs)
        print('lamda:', self.lamda)

        self.lamda += 0.5*(probs - self.delta )
        self.lamda  = T.max(self.lamda.data, T.tensor(0.0))
        
        if self.delta == 1.0:
            self.lamda = T.tensor(0.0)
            # print('lamda:', self.lamda)





        # distribution = self.policy.forward(states)
        # distribution = F.softmax(distribution, dim=1)
        
        # distribution = T.distributions.utils.clamp_probs(distribution)

        # action_probs = T.distributions.Categorical(distribution)
        # action = action_probs.sample()
        # log_probs = action_probs.log_prob(action)
        # print(log_probs)
        log_probs = T.zeros_like(advantages)
        for i in range(len(self.action_memory)):
            log_probs[i] = self.action_memory[i]

        # log_probs = T.tensor(self.action_memory, dtype=T.float).to(self.policy.device)
        
        # print(log_probs)
        # print(self.action_memory)
        # print(advantages)
        # print(-advantages)
        loss = 0
        # print(log_probs,   advantages)
        # self.entropy =self.entropy.squeeze()
        self.entropy_coef *= self.entropy_decay
        loss = (-log_probs*advantages ).mean()
        # print(loss.shape)
        self.entropy =self.entropy.squeeze()
        # self.entropy = -(T.exp(log_probs) * log_probs).mean()
        # x = x.unsqueeze(-1)
        # print(self.entropy)
        # self.entropy_coef *= self.entropy_decay

        loss -= self.entropy_coef*(self.entropy/20)
        # print(loss)

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.entropy = 0
        return self.lamda.data









        

        

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += (self.reward_memory[k]   * discount)   
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)


        # for commulative costs (constraints) .   C is the summation of th

        C = np.zeros_like(self.cost_memory, dtype=np.float64)
        #C_sum = 0
        for t in range(len(self.cost_memory)):
            C_sum = 0
            discount = 1
            
            for k in range(t, len(self.cost_memory)):
                C_sum +=  int(self.cost_memory[k]) * discount
                # discount *= self.gamma
                # print(C_sum)
                

                
            # print(C_sum ,len(self.cost_memory))
            C[t] = C_sum 
            # print(C_sum)
        # print(C[0])
        # print("================================================")
        C = T.tensor(C, dtype=T.float).to(self.policy.device)


        
        loss = 0

        
        for g ,c,logprob in zip(G,C,self.action_memory):
            # loss += (-g * logprob  - self.lamda*c*logprob*float(C[-1]<= self.threshold))
            loss += -(g * logprob  - self.lamda*logprob*c) #*float(C[0] > self.threshold))
            
            # loss += (-g * logprob )     #+ self.lamda*float(C[0]>= self.threshold)
        

        # loss +=  self.lamda*(float(C[0]>=self.threshold))
        
        loss.backward()
        self.policy.optimizer.step()

    
        # TODO: UPDATE LAMDA
        # print('C[0]' , C[0])
        # print(C[0]>= self.threshold)
        
        # self.lamda  = self.lamda  + 0.1 * (float(C[0]>self.threshold) - self.delta)

        if (C[0] > self.threshold):
            self.lamda  = self.lamda  + 0.5 * (1-self.delta)
        else  :
            self.lamda  = self.lamda  - 0.5 * (self.delta)
        
        
        


        
        
        # self.lamda+= self.lr*((1-self.delta) + float(C[0] >= self.threshold))

        self.lamda  = T.max(self.lamda.data, T.tensor(0.0))
        # self.lamda  = T.min(self.lamda.data, T.tensor(1.5))
        # print(self.lamda)
        


        self.action_memory = []
        self.reward_memory = []
        self.cost_memory = []
        # print(self.lamda.data)
        return self.lamda.data
       


    

    
    
    

    