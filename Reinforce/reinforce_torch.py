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

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        self.cost_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()
        # print(action)
        # print(action.item())
    
    def check_constraints_satisfaction(self):
        C = np.zeros_like(self.cost_memory, dtype=np.float64)
        #C_sum = 0
        for t in range(len(self.cost_memory)):
            C_sum = 0
            discount = 1
            for k in range(t, len(self.cost_memory)):
                C_sum += self.cost_memory[k] 
                discount *= self.gamma
            C[t] = C_sum 
            break
        # print(C[0])
        # print("================================================")
        C = T.tensor(C, dtype=T.float).to(self.policy.device)
        # print(C[0])

        if (C[0]>= 1):
            self.cost_memory = []
            # print(False)

            return False
        self.cost_memory = []
        return True
        

        

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

    def learn(self):
        self.policy.optimizer.zero_grad()

        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []














