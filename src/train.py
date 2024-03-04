from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import os
import torch
import random
from copy import deepcopy
import torch.nn as nn
from evaluate import evaluate_HIV, evaluate_HIV_population


# ======================== Buffer Storage ========================

# Define the replay buffer class that will be used to store the experiences of the agent.
# push : Add a new experience to the buffer
# sample : Sample a batch of experiences from it
# __len__ : Return the number of experiences in the buffer
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

# ======================== Building the DQN Agent ========================

# Define the agent class that will be used to train the agent
# act : Select an action given the current state based on a greedy policy and trained Q function
# save : Save the model to a file
# load : Load the best model 

class ProjectAgent:
    def __init__(self, model_name):
        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.path = os.path.join(os.getcwd(),f'{model_name}.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.0005
        self.Q = self.build_nn([6,128,128,256,256,4]).to(self.device)
        self.Q_target = deepcopy(self.Q).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.gamma = 0.98
        self.batch_size = 800
        self.eps_max = 1
        self.eps_min = 0.01
        self.eps_decay = 19000
        self.eps_delay = 200
        self.eps_step = (self.eps_max-self.eps_min)/self.eps_decay
        self.network_sync_counter = 0 
        self.sync_target_step = 300
    
    # ====== Building the Neural Network function ======
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    # ====== Define the act, save and load functions ======
    def act(self, observation):
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

    # ====== Define the greedy policy ========
    def greedy_policy(self, s):
        with torch.no_grad():
            Q = self.Q(torch.Tensor(s).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    # ====== Define the Optimisation gradient step function ======
        
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
        
        # Compute the loss between the current Q values and the target Q (gain) values
        loss = self.criterion(QXA, target.unsqueeze(1))

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.network_sync_counter += 1

    # ====== Define the training function =========
        
    def train(self, num_episodes, max_episode_steps,learning_rate, eps_delay,gamma,sync_target_step,eps_decay, disable_tqdm=False):
        # Set some variables when resuming training
        self.lr= learning_rate
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.eps_delay = eps_delay
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.sync_target_step = sync_target_step
        self.eps_step = (self.eps_max-self.eps_min)/self.eps_decay
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        epsilon = self.eps_max
        step = 0
        self.best_score_agent = 0
        self.best_score_agent_dr = 0
        for episode in range(num_episodes):
                    print(f'********************** Episode : {episode}; Epsilon : {epsilon} **********************')
                    
                    # Renitialize the environment for different patient states to integrate variability
                    if np.random.rand() < 0.4:
                        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
                    else : 
                        self.env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
                    
                    # Reset the state at the begigning of each episode
                    s, _ = self.env.reset()
                    s = torch.tensor(s, dtype=torch.float32, requires_grad = True).to(self.device)
                    
                    # Perform the training loop for each episode
                    for t in tqdm(range(max_episode_steps), disable=disable_tqdm, position=0, leave=True):
                        
                        # Decay the epsilon value after a certain number of steps
                        if step > self.eps_delay:
                            epsilon = max(self.eps_min, epsilon - self.eps_step)

                        # Select epsilon-greedy action
                        if np.random.rand() < epsilon:
                            a = self.env.action_space.sample()
                        else:
                            self.Q.eval()
                            a = self.greedy_policy(s)
                        
                        # Perform an exploration step in the environment
                        s2, r, _, _, _ = self.env.step(a)
                        s2 = torch.tensor(s2, dtype=torch.float32, requires_grad = True).to(self.device)

                        # Push this exploration to the replay buffer
                        self.replay_buffer.push(s, torch.tensor(a, dtype=torch.long).to(self.device), torch.tensor(r, dtype=torch.float32).to(self.device), s2)
                        episode_cum_reward += r
                        s = s2
                        
                        # Update the target network every sync_target_step steps
                        if  self.network_sync_counter % self.sync_target_step == 0:
                            self.Q_target = deepcopy(self.Q)
                        
                        # Perform a gradient descent step
                        self.gradient_step()
                        
                        step += 1
                        
                        if t ==  max_episode_steps - 1 : 
                            print(f"Cumulated Rewards : {episode_cum_reward}")
                            episode_return.append(episode_cum_reward)
                            episode_cum_reward = 0                            
            
                    # Evaluate the agent after each episode
                    print('3. Evaluating')
                    self.evaluate()
        self.save(self.path)
        return episode_return

    # ====== Define the agent evaluation function =========

    def evaluate(self):
                val_score_agent = evaluate_HIV(agent = self, nb_episode=1)
                print(f"score_agent: {'{:e}'.format(val_score_agent)}")
                val_score_agent_dr: float = evaluate_HIV_population(agent=self, nb_episode=1)
                print(f'score_agent_dr = {val_score_agent_dr:e}')
                if val_score_agent > self.best_score_agent : 
                    self.save('best_score_agent_DQN.pth')
                    print("===== Saved Model for score agent record =====")
                    self.best_score_agent = val_score_agent
                    
                if  val_score_agent_dr > self.best_score_agent_dr :
                    self.save('best_score_agent_dr_DQN.pth')
                    print("===== Saved Model for new agent dr record =====")
                    self.best_score_agent_dr = val_score_agent_dr