#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:49:34 2020

@author: will
"""


import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam,Optimizer
from torch.nn.utils import clip_grad_value_
from torch.distributions import Categorical
from wrappers import make_env, VecEnv
import time
from typing import List
device = 'cuda'


class net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc_V = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        logits = self.fc_policy(conv_out)
        value = self.fc_V(conv_out).squeeze(1)
        return logits,value

    def get_value(self,x):
        with torch.no_grad():
            conv_out = self.conv(x).view(x.size()[0], -1)
            return self.fc_V(conv_out).squeeze(1)
        
        
class A2C_agent(object):
    def __init__(self,net: nn.Module, opt: Optimizer, \
                 gamma: float, clip: float, update_freq=50):
        self.net = net
        self.opt = opt
        self.gamma = gamma
        self.clip = clip
        self.update_freq = update_freq
        self.counter = 0
        
    def act(self,s):
        # call act with overwrite=True at beginging
        logits,value = self.net(s)
        rv = Categorical(logits=logits)
        action = rv.sample()
        return action.cpu().numpy(), rv.log_prob(action), value
    
    def update(self,s_p,r,notDones,log_prob,value):
        value_p = self.net.get_value(s_p)
        delta = r + notDones * self.gamma * value_p - value
        loss_A = torch.mean(-log_prob * delta.detach())
        loss_B = torch.mean(torch.abs(delta))
        loss = loss_A + loss_B
        
        loss.backward()
        clip_grad_value_(self.net.parameters(), self.clip)
        self.counter += 1
        if self.counter % self.update_freq == 0:
            self.opt.step()
            self.opt.zero_grad()
        
def listofnp2torch(x: List[np.ndarray], dtype):
    return torch.tensor(np.array(x),dtype=dtype).to(device)

def env2np(args):
    return [np.array(i,dtype=np.float32) for i in args]

def np2torch(s_next,r,done):
    return torch.tensor(s_next).to(device),torch.tensor(r).to(device),torch.tensor(1-done).to(device)

ENV_NAME = 'PongNoFrameskip-v4'
n_env = 128
LEARNING_RATE = 1e-5
gamma = 0.99
clip = 1.0
max_steps = 100000
report_freq = 500

since = time.time()
env = VecEnv(make_env, ENV_NAME, n=n_env)
model = net(env.observation_space.shape, env.action_space.n).to(device)
opt = Adam(model.parameters(), lr=LEARNING_RATE)    
agent = A2C_agent(model,opt,gamma,clip)

tot_rewards = []
rewards = np.zeros(n_env)
s = listofnp2torch(env.reset(),torch.float32)

for i in range(max_steps):
    action, log_prob, value = agent.act(s)
    s_next,r,done = env2np(env.step(action))
    
    # update rewards tracking
    rewards += r
    done_index = done.astype(np.bool)
    tot_rewards.extend(rewards[done_index])
    rewards[done_index] = 0
    
    s_next,r,notDone = np2torch(s_next,r,done)
    agent.update(s_next, r, notDone, log_prob, value)
    
    s = s_next
    if i % report_freq == 0:
        mean_r = np.mean(tot_rewards[-100:])
        print('steps:{}, episodes:{}, reward:{}'.format(i,len(tot_rewards),mean_r))
        if mean_r > 18: 
            time_elapsed = time.time() - since
            print('Training completed in {} mins with {} episode'.format(time_elapsed/60,len(tot_rewards)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    