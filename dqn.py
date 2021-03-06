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
from wrappers import make_env, VecEnv
import time
from typing import List
# import pdb 
device = 'cuda'

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # BN hurts performance for pong
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

class DuelDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelDQN, self).__init__()
        # BN hurts performance for pong
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_A = nn.Sequential(
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
        A = self.fc_A(conv_out)
        return self.fc_V(conv_out) + A - A.mean(1,keepdim=True)

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
        return torch.tensor(np.array(x)).to(device)

class PrioReplayBuffer:
    def __init__(self, capacity,max_=1e-6):
        self.buffer = []
        self.error = np.ones(capacity,dtype=np.float32) * max_
        self.rewards = np.ones(capacity,dtype=np.float32)
        self.max = max_
        self.pos = 0
        self.capacity = capacity
        
    def __len__(self):
        return len(self.buffer)
        
    def append(self, experience,r):
        self.rewards[self.pos] = np.sqrt(1+r)
        self.error[self.pos] = self.max * self.rewards[self.pos]
        if self.full_capacity():
            self.buffer[self.pos] = experience
        else:
            self.buffer.append(experience)
        self.pos = (self.pos + 1) % self.capacity
        
    def update(self,index,error):
        self.error[index] = error * self.rewards[index]
        self.max = max(self.max,np.max(error))
        
    def sample(self, batch_size,alpha,beta):
        assert self.full_capacity()
        prob = self.error ** alpha
        prob /= prob.sum()
        indices = np.random.choice(self.capacity, batch_size, p=prob)
        weight = (prob[indices]*self.capacity) ** (-beta)
        weight /= weight.max()
        # pdb.set_trace()
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        # data augment
        next_states = [(i[...,::-1] if np.random.rand() < 0.5 else i) for i in next_states]
        return (self.convert(states), self.convert(actions), self.convert(rewards), \
               self.convert(dones), self.convert(next_states), torch.tensor(weight.astype(np.float32)).to(device)), indices

    @staticmethod 
    def convert(x):
        return torch.tensor(np.array(x)).to(device)
        
    def full_capacity(self):
        return len(self) == self.capacity
        
class BaseDQNAgent(object):
    def __init__(self, model: nn.Module, tgt: nn.Module, opt: Optimizer, \
                 action_space: int, gamma: float, clip: float, UseMaxOnly: bool):
        '''
        
        Parameters
        ----------
        model : nn.Module
            DESCRIPTION.
        tgt : nn.Module
            DESCRIPTION.
        opt : Optimizer
            DESCRIPTION.
        action_space : int
            dimention of action space.
        gamma : float
            DESCRIPTION.
        clip : float
            DESCRIPTION.
        UseMaxOnly : bool
            use max q(s,a) or E[q(s,a)], expection is with respect to eps-greedy policy

        Returns
        -------
        None.

        '''
        self.model = model
        self.tgt = tgt
        self.opt = opt
        self.action_space = action_space
        self.gamma = gamma
        self.clip = clip
        self.UseMaxOnly = UseMaxOnly
        
    def copy(self):
        self.tgt.load_state_dict(self.model.state_dict())
        
    def act(self,state: np.ndarray, eps: float) -> np.ndarray:
        # state should be shape (1,...)
        if np.random.rand() < eps:
            out = np.random.randint(0,self.action_space)
            return out
        state = torch.tensor(state[None],dtype=torch.float32).to(device)
        with torch.no_grad():
            # self.model.eval()
            out = self.model(state).max(1)[1]
            # self.model.train()
            return out.item()
    
    def acts(self,states: List[np.ndarray], eps: float) -> np.ndarray:
        # state should be shape (n,...)
        if np.random.rand() < eps:
            return np.random.randint(0,self.action_space,len(states))
        states = torch.tensor(np.array(states),dtype=torch.float32).to(device)
        with torch.no_grad():
            # self.model.eval()
            out = self.model(states).max(1)[1]
            # self.model.train()
            return out.cpu().numpy()

    
class DQNAgent(BaseDQNAgent):       
    def update(self, batch: List[torch.Tensor], eps: float):
        # notDones is one, when done is False, zero otherwise
        states, actions, rewards, notDones, next_states = batch
        with torch.no_grad():
            if self.UseMaxOnly:
                tgts = rewards + notDones * self.gamma * self.tgt(next_states).max(1)[0]
            else:
                n = rewards.shape[0]
                prob = torch.ones(n,self.action_space,dtype=torch.float32,device=device)*eps/self.action_space
                q = self.tgt(next_states)
                max_index = q.max(1)[1]
                prob[range(n),max_index] = eps/self.action_space + 1 - eps
                tgts = rewards + notDones * self.gamma * (q*prob).sum(1)
        yhat = self.model(states).gather(1,actions).squeeze(1)        
        loss = F.smooth_l1_loss(yhat,tgts)
        with amp.scale_loss(loss, self.opt) as scaled_loss:
            scaled_loss.backward()
        clip_grad_value_(amp.master_params(self.opt),self.clip)
        self.opt.step()
        self.opt.zero_grad()

class DoubleDQNAgent(BaseDQNAgent):       
    def update(self, batch: List[torch.Tensor], eps: float):
        # notDones is one, when done is False, zero otherwise
        states, actions, rewards, notDones, next_states, weights = batch
        n = rewards.shape[0]
        with torch.no_grad():
            if self.UseMaxOnly:
                max_index = self.model(next_states).max(1)[1]
                tgts = rewards + notDones * self.gamma * self.tgt(next_states)[range(n),max_index]
            else:
                prob = torch.ones(n,self.action_space,dtype=torch.float32,device=device)*eps/self.action_space
                max_index = self.model(next_states).max(1)[1]
                prob[range(n),max_index] = eps/self.action_space + 1 - eps
                tgts = rewards + notDones * self.gamma * (self.tgt(next_states)*prob).sum(1)
        yhat = self.model(states).gather(1,actions).squeeze(1)        
        loss = F.smooth_l1_loss(yhat,tgts,reduction='none')
        loss_np = loss.detach().cpu().numpy()
        loss = torch.mean(loss * weights)
        with amp.scale_loss(loss, self.opt) as scaled_loss:
            scaled_loss.backward()
        clip_grad_value_(amp.master_params(self.opt),self.clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss_np

def Trainer(ENV_NAME,
            Agent,
            Model,
            Buffer,
            episode=50000,
            GAMMA = 0.99,
            BATCH_SIZE = 32,
            REPLAY_SIZE = 100000,
            LEARNING_RATE = 1e-4,
            SYNC_TARGET_FRAMES = 1000,
            #REPLAY_START_SIZE = 10000,
            EPSILON_DECAY_LAST_FRAME = 5*10**6,
            EPSILON_START = 1.0,
            EPSILON_FINAL = 0.05,
            beta_start = 0.4,
            beta_end = 1.0,
            alpha_start = 0.4,
            alpha_end = 0.8,
            clip = 1.0,
            report_freq=50,
            opt_level='O0',
            UseMaxOnly=True):
    
    # setup
    since = time.time()
    env = make_env(ENV_NAME)
    model = Model(env.observation_space.shape, env.action_space.n).to(device)
    tgt = Model(env.observation_space.shape, env.action_space.n).to(device)
    tgt.eval()
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    model, opt = amp.initialize(model, opt, opt_level=opt_level)
    buffer = Buffer(REPLAY_SIZE)
    agent = Agent(model,tgt,opt,env.action_space.n,GAMMA,clip,UseMaxOnly=UseMaxOnly)
    
    tot_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    REPLAY_START_SIZE = REPLAY_SIZE
    
    for i in range(episode):
        done = False
        rewards = 0 
        s = env.reset()
        while not done:
            # generate experience
            eps = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            alpha = min(alpha_end, alpha_start + frame_idx / EPSILON_DECAY_LAST_FRAME)
            beta = min(beta_end, beta_start + frame_idx / EPSILON_DECAY_LAST_FRAME)
            action = agent.act(s,eps)
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
            batch,index = buffer.sample(BATCH_SIZE,alpha,beta)
            error = agent.update(batch,eps) + 1e-10
            buffer.update(index,error)
            if frame_idx % SYNC_TARGET_FRAMES == 0: agent.copy()
                
        tot_rewards.append(rewards)
        if i%report_freq == 0:
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_r = np.mean(tot_rewards[-report_freq:])
            print('episode:{}, frame:{}, reward:{}, speed:{}'.format(i,frame_idx,mean_r,speed))
            if mean_r > 500: 
                time_elapsed = time.time() - since
                print('Training completed in {} mins with {} episode'.format(time_elapsed/60,i))
                return model

def VecTrainer(ENV_NAME,
                Agent,
                Model,
                Buffer,
                GAMMA = 0.99,
                BATCH_SIZE = 1024,
                REPLAY_SIZE = 100000,
                LEARNING_RATE = 6e-4,
                SYNC_TARGET_FRAMES = 1024,
                #REPLAY_START_SIZE = 10000,
                EPSILON_DECAY_LAST_FRAME = 10**7,
                EPSILON_START = 1.0,
                EPSILON_FINAL = 0.1,
                beta_start = 0.4,
                beta_end = 1.0,
                alpha_start = 0.2,
                alpha_end = 0.8,
                clip = 1.0,
                report_freq=1024*8,
                opt_level='O0',
                UseMaxOnly=True,
                n_env = 32):
    
    
    # setup
    since = time.time()
    env = VecEnv(make_env, ENV_NAME, n=n_env)
    model = Model(env.observation_space.shape, env.action_space.n).to(device)
    tgt = Model(env.observation_space.shape, env.action_space.n).to(device)
    tgt.eval()
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    model, opt = amp.initialize(model, opt, opt_level=opt_level)
    buffer = Buffer(REPLAY_SIZE)
    agent = Agent(model,tgt,opt,env.action_space.n,GAMMA,clip,UseMaxOnly=UseMaxOnly)
    
    tot_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    REPLAY_START_SIZE = REPLAY_SIZE
    s = env.reset()
    rewards = np.zeros(n_env)
    
    while True:
        # generate experience
        eps = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        alpha = min(alpha_end, alpha_start + frame_idx / EPSILON_DECAY_LAST_FRAME)
        beta = min(beta_end, beta_start + frame_idx / EPSILON_DECAY_LAST_FRAME)
        action = agent.acts(s,eps)
        s_next,r,done = env.step(action)
        
        rewards += np.array(r)
        done_index = np.array(done)
        tot_rewards.extend(rewards[done_index])
        rewards[done_index] = 0
        
        for i in range(n_env):
            buffer.append((np.float32(s[i]), \
                           np.array([action[i]],dtype=np.int64),\
                           np.float32(r[i]), \
                           np.float32(1-done[i]), \
                           np.float32(s_next[i])),rewards[i])
        s = s_next
        frame_idx += n_env
        if frame_idx < REPLAY_START_SIZE: continue
            
        # train model
        batch,index = buffer.sample(BATCH_SIZE,alpha,beta)
        error = agent.update(batch,eps) + 1e-10
        buffer.update(index,error)
        if frame_idx % SYNC_TARGET_FRAMES == 0: agent.copy()
            
        if frame_idx % report_freq == 0:
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_r = np.mean(tot_rewards[-500:])
            print('episode:{}, frame:{}, reward:{}, speed:{}'.format(len(tot_rewards),frame_idx,mean_r,speed))
            if mean_r > 100: 
                time_elapsed = time.time() - since
                print('Training completed in {} mins with {} episode'.format(time_elapsed/60,len(tot_rewards)))
                return model
            
def play(env,agent):
    s = env.reset()
    done = False
    tot_r = 0
    while not done:
        a = agent.act(s,0)
        s,r,done,_ = env.step(a)
        tot_r += r
        env.render()
        time.sleep(0.1)
    env.close()
    print('total rewards is :{}'.format(tot_r))
    
if __name__ == "__main__":
    model = VecTrainer('BreakoutNoFrameskip-v4',DoubleDQNAgent,DuelDQN,PrioReplayBuffer,UseMaxOnly=False)
    # if model is not None:
    #     torch.save(model.state_dict(), '/home/will/Desktop/kaggle/RL/dqn.bin')
    