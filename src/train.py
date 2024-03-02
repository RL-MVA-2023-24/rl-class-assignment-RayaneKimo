from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
import numpy as np
import os
import torch
import random
from copy import deepcopy
import torch.nn as nn
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = int(0)

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

    
class ProjectAgent:
    def __init__(self):
        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.path = os.path.join(os.getcwd(),'Best_model.pth')
        self.device = torch.device("cpu")
        print(self.device)
        self.Q = self.build_nn([6,128,128,256,256,4]).to(self.device)
        self.Q_target = deepcopy(self.Q).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.005)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.gamma = 0.90
        self.batch_size = 512
        self.eps_max = 1
        self.eps_min = 0.01
        self.eps_decay = 19000
        self.eps_delay = 400
        self.eps_step = (self.eps_max-self.eps_min)/self.eps_decay
#         self.update_target_tau = 0.001
        self.network_sync_counter = 0 
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)

    def act(self, observation, use_random=False):
        return self.greedy_policy(observation)

    def save(self, path):
        # Save the model state dictionary
        torch.save(self.Q.state_dict(), path)

    def load(self):
        # Load the model state dictionary
        self.Q.load_state_dict(torch.load("best_agent_DQN.pth", map_location=torch.device("cpu")))
        self.Q_target =  deepcopy(self.Q).to(self.device)
        self.Q.eval().to(self.device) 
        self.Q_target.eval().to(self.device)


    def greedy_policy(self, s):
        with torch.no_grad():
            Q = self.Q(torch.Tensor(s).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def gradient_step(self):
    # Ensure we have enough samples in the replay buffer
      if len(self.replay_buffer) > self.batch_size:
        self.Q.train()
        self.Q_target.train()
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Compute Q values for next states and take the max Q value along the actions dimension
        with torch.no_grad():
            QYmax = self.Q_target(next_states).max(1)[0].detach()

        # Compute the target Q values for the current states
        target = rewards + self.gamma * QYmax
        
        # Get the Q values for the chosen actions in the current states
        QXA = self.Q(states).gather(1, actions.to(torch.long).unsqueeze(1))
        
        # Compute the loss between the current Q values and the target Q values
        loss = self.criterion(QXA, target.unsqueeze(1))

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.network_sync_counter += 1


    def train(self, num_episodes, max_episode_steps, disable_tqdm=False):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        epsilon = self.eps_max
        step = 0
        best_score = 0
        for episode in range(num_episodes):
                    print(f'********************** Episode : {episode}; Epsilon : {epsilon} **********************')

                    s, _ = self.env.reset()
                    s = torch.tensor(s, dtype=torch.float32, requires_grad = True).to(self.device)
                    
                    for t in tqdm(range(max_episode_steps), disable=disable_tqdm, position=0, leave=True):
                        
                        if step > self.eps_delay:
                            epsilon = max(self.eps_min, epsilon - self.eps_step)

                        # select epsilon-greedy action
                        if np.random.rand() < epsilon:
                            a = self.env.action_space.sample()
                        else:
                            self.Q.eval()
                            a = self.greedy_policy(s)
                        
                        s2, r, _, _, _ = self.env.step(a)
                        s2 = torch.tensor(s2, dtype=torch.float32, requires_grad = True).to(self.device)

                        # Push this experience to the replay buffer
                        self.replay_buffer.push(s, torch.tensor(a, dtype=torch.long).to(self.device), torch.tensor(r, dtype=torch.float32).to(self.device), s2)
                        episode_cum_reward += r
                        s = s2
                        
                        if  self.network_sync_counter % 400 == 0:
                            self.Q_target = deepcopy(self.Q)
                        
                        self.gradient_step()
                        
                        step += 1
                        
                        if t ==  max_episode_steps - 1 : 
                            print(f"Cumulated Rewards : {episode_cum_reward}")
                            episode_return.append(episode_cum_reward)
                            episode_cum_reward = 0                            
            

                    print('3. Evaluating')
                    best_score = self.evaluate(best_score)
        self.save(self.path)
        return episode_return
