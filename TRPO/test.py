
import torch
import numpy as np 
from collections import namedtuple
import torch.nn.functional as F
rollouts  = []
Rollout = namedtuple('Rollout', ['states',  'next_states', ])

# for i in range (2):
    


#         sample = []

#         for i in range (5):
#             sample.append((np.array(i),np.array(i),np.array(i),np.array(i)))


#         states ,rewards,actions,next_states =zip(*sample)
#         print(states)
#         print("------")
#         states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
#         next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
#         # actions = torch.as_tensor(actions)
#         # rewards = torch.as_tensor(rewards)
#         rollouts.append(Rollout(states, next_states))

#         print("-------")
#         print(states)

# states = torch.cat([r.states for r in rollouts], dim=0)
# #actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()


# print("-----")
# print(states)

# print("printing roollouts")
# print(rollouts)

# for s , next_states in rollouts:
#     print(s)
#     print(next_states[-1])

# delta = 1- 2
# loss = delta **2
# new_state_val = torch.tensor([2]).float()
# print(loss)
# state = torch.tensor([1]).float()
# loss2 = F.mse_loss(state,new_state_val)
# print(loss2)

rewards = [1,2,3,4]
for step in reversed(range(len(rewards))):
    print(rewards[step])


