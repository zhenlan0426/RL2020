#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:26:20 2020

@author: will
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer,Adam
from torch.nn.utils import clip_grad_value_
import numpy as np
from apex import amp
import collections
from wrappers import make_env

device = 'cuda'

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return self.convert(states), self.convert(actions), self.convert(rewards), \
               self.convert(dones), self.convert(next_states)
    
    @staticmethod 
    def convert(self,x):
        return torch.tensor(x).to(device)
    
class Agent(object):
    def __init__(self, model: nn.Module, tgt: nn.Module, opt: Optimizer, action_space: int, gamma: float, clip: float):
        self.model = model
        self.tgt = tgt
        self.opt = opt
        self.action_space = action_space
        self.gamma = gamma
        self.clip = clip
        
    def copy(self):
        self.tgt.load_state_dict(self.model.state_dict())
        
    def act(self,state: torch.Tensor, eps: float) -> np.ndarray:
        # state should be of shape (n,...) and should be on device already
        n = state.shape[0]
        if np.random.rand() < eps:
            out = np.random.randint(0,self.action_space,n)
            return (out[0] if n==1 else out)
        with torch.no_grad():
            out = self.model(state).max(1)[1]
            return out.item() if n==1 else out.cpu().numpy()
    
    def update(self, batch: list[torch.Tensor]):
        # notDones is one, when done is False, zero otherwise
        states, actions, rewards, notDones, next_states = batch
        with torch.no_grad():
            tgts = rewards + notDones * self.gamma * self.tgt(next_states).max(1)[0]
        yhat = self.model(states).gather(1,actions)
        loss = F.smooth_l1_loss(yhat,tgts)
        with amp.scale_loss(loss, self.opt) as scaled_loss:
            scaled_loss.backward()
        clip_grad_value_(amp.master_params(self.opt),self.clip)
        self.opt.step()
        self.opt.zero_grad()


def Trainer(ENV_NAME,
            GAMMA = 0.99,
            BATCH_SIZE = 128,
            REPLAY_SIZE = 100000,
            LEARNING_RATE = 1e-4,
            SYNC_TARGET_FRAMES = 1000,
            REPLAY_START_SIZE = 10000,
            EPSILON_DECAY_LAST_FRAME = 10**5,
            EPSILON_START = 1.0,
            EPSILON_FINAL = 0.02,
            clip = 1.0):
    
    env = make_env(ENV_NAME)
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt = DQN(env.observation_space.shape, env.action_space.n).to(device)
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    model, opt = amp.initialize(model, opt, opt_level="O2")
    
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(model,tgt,opt,env.action_space.n,GAMMA,clip)
    
    # generate experience
    s = env.reset()
    a = agent.act(s)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    