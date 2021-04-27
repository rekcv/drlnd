from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed, buffer_size = int(1e5), batch_size = 64, gamma = 0.99, update_every = 4, tau = 1e-3, learning_rate=5e-4):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # We're solving this problem by using Deep Q Learning. That means...

        # 1) Create local and target networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # 2) Initialize replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma, tau)

    def learn(self, experiences, gamma, tau):
        states, actions, rewards, next_states, dones = experiences

        # 1) Get predicted Q targets
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # 2) Calculate Q targets
        Q_targets = self.qnetwork_local(states).gather(1, actions)

        # 3) Get expected Q from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 

        # 4) Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # 5) And minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) Finally, update target network with the weights calculated so far for the local network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))