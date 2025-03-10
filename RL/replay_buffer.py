# import random 
# import torch 
# import numpy as np

# class ReplayBufferDQN:
#     def __init__(self, buffer_size:int, seed:int = 42):
#         self.buffer_size = buffer_size
#         self.seed = seed
#         self.buffer = []
#         random.seed(self.seed)
    
#     def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray
#             , done:bool):
#         """Add a new experience to the buffer

        

#         Args:
#             state (np.ndarray): the current state of shape [n_c,h,w]
#             action (int): the action taken
#             reward (float): the reward received
#             next_state (np.ndarray): the next state of shape [n_c,h,w]
#             done (bool): whether the episode is done
#         """
#         # ====== TODO: ======

#         # experience = (state, action, reward, next_state, done)
#         # if len(self.buffer) >= self.buffer_size:
#         #     self.buffer.pop(0)
#         # self.buffer.append(experience)

#         experience = (state, action, reward, next_state, done)
#         self.buffer.append(experience)
#         if len(self.buffer) > self.buffer_size:
#             self.buffer.pop(0)

        
    
#     def sample(self,batch_size:int,device = 'cpu'):
#         """Sample a batch of experiences from the buffer

#         Args:
#             batch_size (int): the number of samples to take

#         Returns:
#             states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
#             actions (torch.Tensor): a np.ndarray of shape [batch_size] of dtype int64
#             rewards (torch.Tensor): a np.ndarray of shape [batch_size] of dtype float32
#             next_states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
#             dones (torch.Tensor): a np.ndarray of shape [batch_size] of dtype bool
#         """
#         # ====== TODO: ======

#         idx = random.sample(range(len(self.buffer)), batch_size)

#         states, actions, rewards, next_states, dones = [],[],[],[],[]

#         for i in idx:
#           state, action, reward, next_state, done = self.buffer[i]
#           states.append(torch.from_numpy(state))
#           actions.append(action)
#           rewards.append(reward)
#           next_states.append(torch.from_numpy(next_state))
#           dones.append(done)

#         states = torch.stack(states).to(device).float()
#         actions = torch.tensor(actions).to(device)
#         rewards = torch.tensor(rewards).to(device).float()
#         next_states = torch.stack(next_states).to(device).float()
#         dones = torch.tensor(dones).to(device)

#         # states = torch.tensor(states).to(device).float()
#         # actions = torch.tensor(actions).to(device).long()
#         # rewards = torch.tensor(rewards).to(device).float()
#         # next_states = torch.tensor(next_states).to(device).float()
#         # dones = torch.tensor(dones).to(device).bool()

#         return states, actions, rewards, next_states, dones





#         # Randomly select batch_size indices
#         # indices = random.sample(range(len(self.buffer)), batch_size)

#         # states = [self.buffer[idx][0] for idx in indices]
#         # actions = [self.buffer[idx][1] for idx in indices]
#         # rewards = [self.buffer[idx][2] for idx in indices]
#         # next_states = [self.buffer[idx][3] for idx in indices]
#         # dones = [self.buffer[idx][4] for idx in indices]



#         # states = torch.tensor(states).to(device).float()
#         # actions = torch.tensor(actions).to(device).long()
#         # rewards = torch.tensor(rewards).to(device).float()
#         # next_states = torch.tensor(next_states).to(device).float()
#         # dones = torch.tensor(dones).to(device).bool()

#         # # states = states.clone().detach().to(self.device).float()
#         # # actions = actions.clone().detach().to(self.device).long()
#         # # rewards = rewards.clone().detach().to(self.device).float()
#         # # next_states = next_states.clone().detach().to(self.device).float()
#         # # dones = dones.clone().detach().to(self.device).bool()

#         # return states, actions, rewards, next_states, dones
    
#     def __len__(self):
#         return len(self.buffer)



import random 
import torch 
import numpy as np

class ReplayBufferDQN:
    def __init__(self, buffer_size:int, seed:int = 42):
        self.buffer_size = buffer_size
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)
    
    def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray
            , done:bool):
        """Add a new experience to the buffer

        Args:
            state (np.ndarray): the current state of shape [n_c,h,w]
            action (int): the action taken
            reward (float): the reward received
            next_state (np.ndarray): the next state of shape [n_c,h,w]
            done (bool): whether the episode is done
        """
        self.buffer.append((state,action,reward,next_state,done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
    
    def sample(self,batch_size:int,device = 'cpu'):
        """Sample a batch of experiences from the buffer

        Args:
            batch_size (int): the number of samples to take

        Returns:
            states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            actions (torch.Tensor): a np.ndarray of shape [batch_size] of dtype int64
            rewards (torch.Tensor): a np.ndarray of shape [batch_size] of dtype float32
            next_states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            dones (torch.Tensor): a np.ndarray of shape [batch_size] of dtype bool
        """
        idx = random.sample(range(len(self.buffer)),batch_size)

        states,actions,rewards,next_states,dones = [],[],[],[],[]

        for i in idx:
            state,action,reward,next_state,done = self.buffer[i]
            states.append(torch.from_numpy(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.from_numpy(next_state))
            dones.append(done)
        
        states = torch.stack(states).to(device).float()
        actions = torch.tensor(actions).to(device).long()
        rewards = torch.tensor(rewards).to(device).float()
        next_states = torch.stack(next_states).to(device).float()
        dones = torch.tensor(dones).to(device).bool()
        return states,actions,rewards,next_states,dones


    def __len__(self):
        return len(self.buffer)
