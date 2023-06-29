import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from 02_policy_gradient_exercise import PG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticNet(nn.Module):
    
    def __init__(self, hidden_dim=16) -> None:
        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        s = self.hidden(s)
        s = F.relu(s)
        s = self.output(s)
        return s

if __name__ == "__main__":
    actor = PG()
    critic_net = CriticNet().to(device)
    for i in range(1500):
        done = False
        s, _ = actor.env.reset()
        states, actions, rwds = [],[],[]
        while not done:
            states.append(s)
            a = actor.pick_action(s)
            actions.append(a)
            s, r, trunc, term, _ = actor.env.step(a)
            rwds.append(r)
        cum_rwds = actor.compute_cumulative_rwd(rwds)
        # Opti critic
        opti_critic = torch.optim.AdamW(critic_net.parameters(),lr=0.001)
        values = critic_net(s).squeeze(dim=1)
        opti_critic.zero_grad()
        critic_loss = F.mse_loss(input=values,target=cum_rwds)
        critic_loss.backward()
        opti_critic.step()
        # Opti actor
        opti_actor = torch.optim.AdamW(critic_net.parameters(),lr=0.001)
        states = torch.tensor(states,dtype=torch.float).to(device)
        actions = torch.tensor(actions,dtype=torch.float).to(device)
        cum_rwds = torch.tensor(cum_rwds,dtype=torch.float).to(device)
        logits = actor.policy_pi(states)
        log_probs = -F.cross_entropy(input=logits,target=actions)
        actor_loss = -log_probs*(cum_rwds-values)
        opti_actor.zero_grad()
        actor_loss.sum.backward()
        opti_actor.step()
    actor.env.step()
            