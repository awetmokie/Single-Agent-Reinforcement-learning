import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

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
    


class CostCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims):
        super(CostCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

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
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4 ,lamda = 2 ,threshold = 0.7 ,delta = 0.1) :
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.buffer_memory = []
        self.action_memory = []
        self.cost_memory = []
        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)
        self.cost_critic = CostCriticNetwork(0.0005, input_dims)
        self.lamda =  T.tensor(0.0, requires_grad=True).to(self.policy.device)
        self.threshold = threshold
        self.delta =delta

        

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item(),log_probs

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def store_costs(self, cost):
        self.cost_memory.append(cost)
    

    def save_model(self,path= "state_dict_model.pt"):
        # Specify a path
        self.policy.save_model(self.policy,path)
        self.cost_critic.save_model(self.cost_critic,path="state_dict_model_cost.pt")
    
    def load_model(self ,path="state_dict_model.pt"):

        self.policy.load_model(path)
        self.cost_critic.load_model(path="state_dict_model_cost.pt")
    

    def check_constraints_satisfaction(self):
        C = np.zeros_like(self.cost_memory, dtype=np.float64)
        #C_sum = 0
        counter = 0 
        average_violation =  0.0
        for t in range(len(self.cost_memory)):
            C_sum = 0
            discount = 1
            
            for k in range(t, len(self.cost_memory)):
                C_sum +=  self.cost_memory[k]* discount
                discount *= self.gamma

            
            # print(C_sum)
                
            
            if (C_sum > self.threshold):
                counter+=1
           
           
        
        C = T.tensor(C, dtype=T.float).to(self.policy.device)
        
        average_violation = float(counter)/len(self.cost_memory)
        self.cost_memory = []

        if (average_violation > self.delta):
            return True
        return False

        
    

    def learn(self):
        self.policy.optimizer.zero_grad()
        self.cost_critic.optimizer.zero_grad()


        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}


        # comulative reward

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += (self.reward_memory[k]  ) * discount 
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)


        # for commulative costs (constraints) .   C is the summation of th

        C = np.zeros_like(self.cost_memory, dtype=np.float64)
        #C_sum = 0
        counter = 0 
        average_violation =  0.0
        for t in range(len(self.cost_memory)):
            C_sum = 0
            discount = 1
            
            for k in range(t, len(self.cost_memory)):
                C_sum +=  self.cost_memory[k]* discount
                discount *= self.gamma

            
            # print(C_sum)
                
            
            if (C_sum > self.threshold):
                counter+=1
           
           
        
        C = T.tensor(C, dtype=T.float).to(self.policy.device)

        average_violation = float(counter)/len(self.cost_memory)
        
        loss = 0

        # print(C[0])
        for g ,c,logprob in zip(G, C,self.action_memory):
            # loss += (-g * logprob  - self.lamda*c*logprob*float(C[-1]<= self.threshold))
            # loss += (-g * logprob  - self.lamda*logprob*float(C[0]>= self.threshold))
            
            loss += (-g * logprob ) + self.lamda *logprob*float(c > self.threshold)

        
        loss.backward()
        self.policy.optimizer.step()

    
        # TODO: UPDATE LAMDA
        print('average_violation',average_violation)
        
        self.lamda  = self.lamda  + 0.02 * (average_violation - self.delta)
        
        


        
        
        # self.lamda+= self.lr*((1-self.delta) + float(C[0] >= self.threshold))

        self.lamda  = T.max(self.lamda.data, T.tensor(0.0).to(self.policy.device))
        # self.lamda  = T.min(self.lamda.data, T.tensor(1.5))
        print(self.lamda)
       

        self.action_memory = []
        self.reward_memory = []
        self.cost_memory = []

    

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
    

    def compute_risks(self,next_value, rewards):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * 1
            returns.insert(0, R)
        return returns
    


    def get_retuns(self):


        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += (self.reward_memory[k]  ) * discount 
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)



        loss = 0

        # print(C[0])
        for g ,logprob in zip(G,self.action_memory):
            # loss += (-g * logprob  - self.lamda*c*logprob*float(C[-1]<= self.threshold))
            # loss += (-g * logprob  - self.lamda*logprob*float(C[0]>= self.threshold))
            
            loss += (-g * logprob ) 
        
        self.action_memory = []
        self.reward_memory = []
        self.cost_memory = []

        

        
        return loss


    def updated_learn(self ,samples):
        

        self.policy.optimizer.zero_grad()
        self.cost_critic.optimizer.zero_grad()
        # Transpose our samples
        #states, actions, rewards, next_states = zip(*samples)
        
        states_values  = []
        next_state_values = []
        rewards  = []
        log_probs = []
        advantages =[]  #requires_grad=True
        last_state = None
        intial_risk = T.tensor(0.0).to(self.cost_critic.device)
        count = 0 
        for state ,action , reward , cost,next_state ,done in samples:
            state = T.tensor([state], dtype=T.float).to(self.cost_critic.device)
            next_state = T.tensor([next_state], dtype=T.float).to(self.cost_critic.device)
            critic_value = self.cost_critic.forward(state)
            critic_value_ = self.cost_critic.forward(next_state)
            if count == 0 :
                intial_risk = critic_value 
            count += 1
            delta = cost + critic_value_*(1-int(done)) - critic_value # might of the correct interpretation
            
            advantages.append(delta)
            last_state = critic_value_
            
            rewards.append(T.tensor([reward], dtype=T.float).to(self.policy.device))
            log_probs.append(action)
            states_values.append(critic_value)
            next_state_values.append(critic_value)
        # print("---------- advantages -----------")
        

        returns =self.compute_returns(last_state , rewards)
        
        returns = T.cat(returns).detach()
        states_values = T.cat(states_values)
        # print("----------   return----------")
        # print(returns)
        
        # advantages = returns - self.lamda*states_values
        
        advantages = T.cat(advantages)
        
        log_probs = T.cat(log_probs)
        
        
        # print("----------advantages--------")
        
        # print(advantages)
        
        # actor_loss = -(log_probs * returns.detach()).mean()
        actor_loss = self.get_retuns()

        # print(intial_risk.data)

        if intial_risk.data > self.delta:
            cost_advatage =self.lamda*log_probs*advantages
            cost_advatage = cost_advatage.sum()
            
            actor_loss = actor_loss  + cost_advatage
            # actor_loss = actor_loss.sum()
            self.lamda  = self.lamda  + 0.02 * (intial_risk - self.delta)


        actor_loss.backward(retain_graph = True)
        self.policy.optimizer.step()
        

        critic_loss =  .5 * (advantages.pow(2)).mean()

       
        (critic_loss).backward(retain_graph = True)
        self.cost_critic.optimizer.step()

        self.lamda  = T.max(self.lamda.data, T.tensor(0.0).to(self.policy.device))

        return intial_risk.item()



    

    
    
    

    