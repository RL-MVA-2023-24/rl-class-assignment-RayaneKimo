from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
import numpy as np
from tqdm import tqdm
import random
import os
import torch
import pickle
from copy import deepcopy
import torch.optim as optim
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


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define your neural network structure here
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
class ProjectAgent:
    def __init__(self):
        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.path = os.path.join(os.getcwd(),'best_agent.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = DQN(6, 4).to(self.device)
        self.Q_target = deepcopy(self.Q)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.gamma = 0.95
        self.batch_size = 512
        self.eps_max = 1
        self.eps_min = 0.01
        self.eps_decay = 18000
        self.eps_delay = 15
        self.eps_step = (self.eps_max-self.eps_min)/self.eps_decay
        self.update_target_tau = 0.001

    def act(self, observation, use_random=False):
        return self.greedy_policy(observation)

    def save(self, path):
        # Save the model state 
        torch.save(self.Q.state_dict(), path)

    def load(self):
        # Load the model state 
        self.Q.load_state_dict(torch.load('best_agent.pth'), map_location=self.device)
        self.Q.eval().to(self.device) 
        self.Q_target =  deepcopy(self.Q).to(self.device)


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

    def evaluate(self, best_score):
                val_score = evaluate_HIV(agent = self, nb_episode=1)
                print(f" == End of episode. Score: {'{:e}'.format(val_score)} ==")
                if val_score > best_score:
                    self.save('best_agent.pth')
                    return val_score
                return best_score
    
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
                        
                        # Explore Once
                        s2, r, _, _, _ = self.env.step(a)
                        s2 = torch.tensor(s2, dtype=torch.float32, requires_grad = True).to(self.device)

                        # Push this experience to the replay buffer
                        self.replay_buffer.push(s, torch.tensor(a, dtype=torch.long).to(self.device), torch.tensor(r, dtype=torch.float32).to(self.device), s2)
                        episode_cum_reward += r
                        
                        # Make a gradient step with random sampling
                        self.gradient_step()
                        
                        # Update target
                        target_state_dict = self.Q_target.state_dict()
                        model_state_dict = self.Q.state_dict()
                        tau = self.update_target_tau
                        for key in model_state_dict:
                            target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                        self.Q_target.load_state_dict(target_state_dict)
                        
                        step += 1
                        
                        if t ==  max_episode_steps - 1 : 
                            print(f"Cumulated Rewards : {episode_cum_reward}")
                            episode_return.append(episode_cum_reward)
                            episode_cum_reward = 0
                        else : 
                            s = s2
            

                    print('3. Evaluating')
                    best_score = self.evaluate(best_score)
        self.save(self.path)
        return episode_return
