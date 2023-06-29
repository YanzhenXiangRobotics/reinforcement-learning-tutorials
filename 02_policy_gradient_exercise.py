import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PI(nn.Module):

    def __init__(self, hidden_dim=64) -> None:
        self.hidden = nn.Linear(4,hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        s = self.hidden(s)
        s = F.relu(s)
        s = self.output(s)
        return s

class PG():

    def __init__(self) -> None:
        self.env = gym.make("CartPole-v1")
        self.policy_pi = PI().to(device)
        self.gamma = 0.99

    def pick_action(self, s):
        with torch.no_grad():
            s = np.expand_dims(s, axis=0)
            s = torch.tensor(s, dtype=torch.float).to(device)
            logits = self.policy_pi(s)
            logits = torch.squeeze(logits, dim=0)
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(input=probs,num_samples=1)
            return a.to_list()[0]
    
    def compute_cumulate_rwd(self, rwds):
        cum_rwds = np.copy(rwds)
        for i in reversed(range(len(rwds))):
            cum_rwds[i] += self.gamma * cum_rwds[i+1] if (i+1<len(rwds)) else 0
        return cum_rwds
    
if __name__ == "__main__":
    pg_agent = PG()
    opti = torch.optim.AdamW(pg_agent.policy_pi.parameters(),lr=-0.001)
    rwd_records = []
    for i in range(1000):
        # rollout policy
        states, actions, rwds = [], [], []
        s, _ = pg_agent.env.reset()
        done = False
        while not done:
            states.append(s)
            a = pg_agent.pick_action(s)
            s, r, term, trunc, _ = pg_agent.env.step(a)
            actions.append(a)
            rwds.append(r)
        cum_rwds = pg_agent.compute_cumulate_rwd(rwds)
        # training
        states = torch.tensor(states,dtype=torch.float).to(device)
        actions = torch.tensor(actions,dtype=torch.float).to(device)
        cum_rwds = torch.tensor(cum_rwds,dtype=torch.float).to(device)
        opti.zero_grad()
        logits = pg_agent.policy_pi(states)
        log_probs = -F.cross_entropy(input=logits,target=actions)
        loss = -cum_rwds*log_probs
        loss.sum().backward()
        opti.step()
        rwd_records.append(rwds)
    pg_agent.env.close()