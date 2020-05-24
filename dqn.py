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
import time
from typing import List
#import pdb 
device = 'cuda'

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
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
    def convert(x):
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
        
    def act(self,state: np.ndarray, eps: float) -> np.ndarray:
        # state should be shape (1,...)
        if np.random.rand() < eps:
            out = np.random.randint(0,self.action_space)
            return out
        state = torch.tensor(state[None],dtype=torch.float32).to(device)
        with torch.no_grad():
            self.model.eval()
            out = self.model(state).max(1)[1]
            self.model.train()
            return out.item()
    
    def acts(self,states: np.ndarray, eps: float) -> np.ndarray:
        # state should be shape (n,...)
        if np.random.rand() < eps:
            return np.random.randint(0,self.action_space,states.shape[0])
        states = torch.tensor(states,dtype=torch.float32).to(device)
        with torch.no_grad():
            self.model.eval()
            out = self.model(states).max(1)[1]
            self.model.train()
            return out.cpu().numpy()
        
    def update(self, batch: List[torch.Tensor]):
        # notDones is one, when done is False, zero otherwise
        states, actions, rewards, notDones, next_states = batch
        #pdb.set_trace()
        with torch.no_grad():
            tgts = rewards + notDones * self.gamma * self.tgt(next_states).max(1)[0]
        yhat = self.model(states).gather(1,actions).squeeze(1)        
        loss = F.smooth_l1_loss(yhat,tgts)
        with amp.scale_loss(loss, self.opt) as scaled_loss:
            scaled_loss.backward()
        clip_grad_value_(amp.master_params(self.opt),self.clip)
        self.opt.step()
        self.opt.zero_grad()


def Trainer(ENV_NAME,
            episode=1000,
            GAMMA = 0.99,
            BATCH_SIZE = 128,
            REPLAY_SIZE = 100000,
            LEARNING_RATE = 1e-4,
            SYNC_TARGET_FRAMES = 1000,
            REPLAY_START_SIZE = 10000,
            EPSILON_DECAY_LAST_FRAME = 10**5,
            EPSILON_START = 1.0,
            EPSILON_FINAL = 0.02,
            clip = 1.0,
            report_freq=10):
    
    # setup
    since = time.time()
    env = make_env(ENV_NAME)
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt.eval()
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    model, opt = amp.initialize(model, opt, opt_level="O2")
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(model,tgt,opt,env.action_space.n,GAMMA,clip)
    
    tot_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
            
    for i in range(episode):
        done = False
        rewards = 0 
        s = env.reset()
        while not done:
            # generate experience
            eps = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            action = agent.act(s,eps)
            #pdb.set_trace()
            s_next,r,done,_ = env.step(action)
            rewards += r
            buffer.append((np.float32(s), \
                           np.array([action],dtype=np.int64),\
                           np.float32(r), \
                           np.float32(1-done), \
                           np.float32(s_next)))
            s = s_next
            frame_idx += 1
            if frame_idx < REPLAY_START_SIZE: continue
                
            # train model
            batch = buffer.sample(BATCH_SIZE)
            agent.update(batch)
            if frame_idx % SYNC_TARGET_FRAMES == 0: agent.copy()
                
        tot_rewards.append(rewards)
        if i%report_freq == 0:
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_r = np.mean(tot_rewards[-report_freq:])
            print('episode:{}, frame:{}, reward:{}, speed:{}'.format(i,frame_idx,mean_r,speed))
            if mean_r > 18: 
                time_elapsed = time.time() - since
                print('Training completed in {} mins with {} episode'.format(time_elapsed/60),i)
                return model
            
    
if __name__ == "__main__":
    Trainer('PongNoFrameskip-v4')    
    
    