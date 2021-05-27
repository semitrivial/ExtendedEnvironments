import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

from util import memoize

class DummyEnv:
    def set_meta(self, num_legal_actions, num_possible_obs):
        self.num_legal_actions = num_legal_actions
        self.num_possible_obs = num_possible_obs
    def set_rewards_and_observs(self, rewards, observs):
        self.rewards = rewards
        self.observs = observs
        self.history = []
        self.i = 1
    def get_state(self):
        return self.observs[0]
    def act(self, action, network_act):
        try:
            reward = self.rewards[self.i]
            obs = self.observs[self.i]
        except Exception:
            import pdb; pdb.set_trace()
        self.i += 1
        self.history.append([obs, action, reward])
        return reward, obs, self.history

cache_custom_DQN = {}

@memoize
def custom_DQN_agent(prompt, num_legal_actions, num_possible_obs, **kwargs):
    meta = (num_legal_actions, num_possible_obs)
    num_observs = (len(prompt)+1)/3
    train_on_len = 3*pow(2, int(math.log2(num_observs)))-1
    train_on = prompt[:train_on_len]

    if not('lookback' in kwargs):
        kwargs['lookback'] = 10

    if not((train_on, meta) in cache_custom_DQN):
        rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
        observs = [train_on[i+1] for i in range(0,train_on_len,3)]
        dummy_env = DummyEnv()
        dummy_env.set_meta(num_legal_actions, num_possible_obs)
        dummy_env.set_rewards_and_observs(rewards, observs)

        A = RecurrentAgent(
            network=TreasureGRUNet,
            game_env=dummy_env,
            **kwargs
        )

        A.train_on_history(train_prompt=train_on)
        cache_custom_DQN[(train_on, meta)] = A
    else:
        A = cache_custom_DQN[(train_on, meta)]

    state_obs = [prompt[-1]]
    filled_prompt = [0]*(3*kwargs['lookback']) + list(prompt[1:-1])
    state = (filled_prompt + state_obs)[-(3*kwargs['lookback']+1):]

    state = torch.tensor(state, dtype=torch.float, device=device)
    state = state.reshape((1,-1,1))

    return A.blind_act(state)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity, ), dtype=np.float32)

    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_prio

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


class TreasureGRUNet(nn.Module):
    def __init__(self,hidden_dim=5, n_layers=2, dropout_p=0.2):
        super(TreasureGRUNet, self).__init__()
        input_dim = 1
        output_dim = 2
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class RecurrentAgent:
    def __init__(self, network, game_env, lookback, BATCH_SIZE=128, GAMMA=0.999, EPS_START=1, EPS_END=0.05,EPS_DECAY=1000,TARGET_UPDATE=100, learning_rate=1e-1, prioritized=False):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY=10000
        self.TARGET_UPDATE=TARGET_UPDATE
        self.learning_rate = learning_rate

        self.q_net = network().to(device)

        self.target_net = network().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.SGD(self.q_net.parameters(), self.learning_rate)

        self.prioritized = prioritized
        if prioritized:
            self.replay_memory = PrioritizedReplayMemory(10_000)
        else:
            self.replay_memory = ReplayMemory(10_000)

        self.steps_done = 0
        self.losses = []
        self.episodes = 0
        self.scores = []

        self.game_env = game_env

        self.state = None
        self.history = None

        self.lookback = lookback

        beta_start = 0.4
        beta_frames = 10000
        self.beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    def get_initial_state(self):
        self.state = self.game_env.get_state()

    def create_network_state(self,state, history):
        if history is not None:
            sequence = [e for et in history for e in et]
        else:
            sequence = []
        sequence += [state]

        lookback_window = (3 * self.lookback) + 1
        if len(sequence) < lookback_window:
            sequence = [0] * lookback_window + sequence
        available_seq = sequence[-lookback_window:]
        seq_state = torch.tensor(available_seq, dtype=torch.float, device=device).reshape((1,-1,1))

        return seq_state

    def random_act(self):
        return np.random.randint(2)

    def network_act(self,state,history=None):
        if not isinstance(state,torch.Tensor):
            state = self.create_network_state(state=state, history=history)

        assert state.shape == torch.Size([1,((self.lookback * 3) + 1),1])
        self.q_net.eval()
        with torch.no_grad():
            h = self.q_net.init_hidden(state.shape[0])
            action_values, h = self.q_net(state, h)
            result = action_values.max(1)[1].unsqueeze(1)
            assert result.shape == torch.Size([1,1])
        self.q_net.train()

        return result.item()

    def blind_act(self,state):
        assert state.shape == torch.Size([1,((self.lookback * 3) + 1),1])
        self.q_net.eval()
        with torch.no_grad():
            h = self.q_net.init_hidden(state.shape[0])
            action_values, h = self.q_net(state, h)
            result = action_values.max(1)[1].unsqueeze(1)
            assert result.shape == torch.Size([1,1])

        return result.item()

    def calculate_epsilon_threshold(self):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        eps_sample = np.random.rand()

        return eps_threshold, eps_sample

    def act(self):
        if self.state is None:
            self.get_initial_state()
            self.state = self.create_network_state(state=self.state, history=self.history)
        eps_threshold, eps_sample = self.calculate_epsilon_threshold()

        self.steps_done += 1

        if eps_sample > eps_threshold:
            action = self.network_act(self.state)
        else:
            action = self.random_act()

        assert isinstance(action, int)

        reward, next_state, self.history = self.game_env.act(action, self.network_act)

        self.action = torch.tensor([action], device=device).unsqueeze(0)
        self.reward = torch.tensor([reward], device=device).unsqueeze(0)
        self.next_state = self.create_network_state(state = next_state, history = self.history)

        return reward

    def sample_memory(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return

        transitions = self.replay_memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        return batch

    def sample_memory_prioritized(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return None, None, None

        beta = self.beta_by_frame(self.steps_done)
        transitions, indices, weights = self.replay_memory.sample(self.BATCH_SIZE, beta)
        batch = Transition(*zip(*transitions))

        return batch, indices, weights

    def optimize(self):
        batch = self.sample_memory()
        if batch is None:
            return

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        assert state_batch.shape[0] == self.BATCH_SIZE
        assert action_batch.shape[0] == self.BATCH_SIZE
        assert reward_batch.shape[0] == self.BATCH_SIZE
        assert next_state_batch.shape[0] == self.BATCH_SIZE

        h = self.q_net.init_hidden(self.BATCH_SIZE).data
        q_values, h = self.q_net(state_batch, h)
        assert q_values.shape[0] == self.BATCH_SIZE

        q_values = q_values.gather(1, action_batch)


        next_h = self.target_net.init_hidden(self.BATCH_SIZE).data
        next_state_action_values, next_h = self.target_net(next_state_batch, next_h)
        next_state_action_values = next_state_action_values.max(1)[0].unsqueeze(1).detach()

        assert next_state_action_values.shape == torch.Size([self.BATCH_SIZE,1])

        expected_q_values = (next_state_action_values * self.GAMMA) + reward_batch

        q_loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.losses.append(q_loss)

        self.optimizer.zero_grad()
        q_loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer.step()

    def optimize_prioritized(self):

        batch, indices, weights = self.sample_memory_prioritized()
        if batch is None:
            return

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        weights = np.expand_dims(weights, 1)
        weights_batch = torch.tensor(weights, dtype=torch.float,device=device)

        assert state_batch.shape[0] == self.BATCH_SIZE
        assert action_batch.shape[0] == self.BATCH_SIZE
        assert reward_batch.shape[0] == self.BATCH_SIZE
        assert next_state_batch.shape[0] == self.BATCH_SIZE

        h = self.q_net.init_hidden(self.BATCH_SIZE).data
        q_values, h = self.q_net(state_batch, h)
        assert q_values.shape[0] == self.BATCH_SIZE

        q_values = q_values.gather(1, action_batch)


        next_h = self.target_net.init_hidden(self.BATCH_SIZE).data
        next_state_action_values, next_h = self.target_net(next_state_batch, next_h)
        next_state_action_values = next_state_action_values.max(1)[0].unsqueeze(1).detach()

        assert next_state_action_values.shape == torch.Size([self.BATCH_SIZE,1])

        expected_q_values = (next_state_action_values * self.GAMMA) + reward_batch

        q_loss = F.smooth_l1_loss(q_values, expected_q_values)
        q_loss = (q_values - expected_q_values).pow(2) * weights_batch
        prios = q_loss + 1e-5
        q_loss = q_loss.mean()
        self.losses.append(q_loss)

        self.optimizer.zero_grad()
        q_loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        self.replay_memory.update_priorities(indices, prios.data.cpu().numpy())

    def play(self, training=True, episodes = 1_000):
        for i_episode in range(episodes):
            score = self.act()
            self.scores.append(score)

            if training:
                self.replay_memory.push(self.state, self.action, self.next_state, self.reward)
                if self.prioritized:
                    self.optimize_prioritized()
                else:
                    self.optimize()

                if i_episode % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

            self.state = self.next_state
            self.next_state = None

    def train_on_history(self, train_prompt):
        oar_style_prompt = train_prompt[1:]
        replays = math.ceil(len(oar_style_prompt) / 3)
        history = [oar_style_prompt[i * 3: (i+1) * 3] for i in range(replays)]
        for i in range(len(history) - 1):
            state = self.create_network_state(state=history[i][0],history=history[:i])
            next_state = self.create_network_state(state=history[i+1][0],history=history[:i+1])
            action = torch.tensor([history[i][1]], device=device).unsqueeze(0)
            reward = torch.tensor([history[i][2]], device=device).unsqueeze(0)

            self.replay_memory.push(state, action, next_state, reward)
            if self.prioritized:
                self.optimize_prioritized()
            else:
                self.optimize()

            if i % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())




