import numpy as np
import gym
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1
        self.epsilon_decay = self.epsilon / 4000
        self.env = gym.make("CartPole-v1")

        self.s_dis_dim = (20,20,20,20)
        self.q_table = np.zeros(self.s_dis_dim+(self.env.action_space.n,))

        self.bins = []
    
    def make_bins(self):
        for i in range(len(self.s_dis_dim)):
            bin_item = np.linspace(start=self.env.observation_space.low[i] if (i == 0) or (i == 2) else -4,
                                    stop = self.env.observation_space.high[i] if (i == 0) or (i == 2) else 4,
                                    num = self.s_dis_dim[i])
            bin_item = np.delete(bin_item,0)
            self.bins.append(bin_item)

    def discretize_state(self, s):
        s_dis = np.zeros_like(s)
        for i, s_item in enumerate(s):
            s_dis[i] = np.digitize(s_item, self.bins[i])
        return s_dis.tolist()

    def pick_action(self, s_dis):
        if np.random.random() > self.epsilon:
            return np.argmax(self.q_table[tuple(s_dis)])
        else:
            return np.random.randint(0, self.env.action_space.n)
        
    def plot_rwds(self, rwds):
        average_reward = []
        for idx in range(len(rwds)):
            avg_list = np.empty(shape=(1,), dtype=int)
            if idx < 50:
                avg_list = rwds[:idx+1]
            else:
                avg_list = rwds[idx-49:idx+1]
            average_reward.append(np.average(avg_list))
        # Plots
        plt.plot(rwds)
        plt.plot(average_reward)

if __name__ == "__main__":
    ql_agent = QLearning()
    ql_agent.make_bins()
    rwds = []
    for i in range(6000):
        rwd_per_ep = 0
        s, _ = ql_agent.env.reset()
        done = False
        s_dis = ql_agent.discretize_state(s)
        while not done:
            a = ql_agent.pick_action(s_dis)
            s, r, term, truc, _ = ql_agent.env.step(a)
            s_dis_next = ql_agent.discretize_state(s)

            maxQ = np.max(ql_agent.q_table[tuple(s_dis_next)])
            ql_agent.q_table[tuple(s_dis)][a] += ql_agent.alpha*(r+maxQ-ql_agent.q_table[tuple(s_dis)][a])

            s_dis = s_dis_next

            rwd_per_ep += r
        ql_agent.epsilon = np.max(0, ql_agent.epsilon - ql_agent.epsilon_decay)
        rwds.append(rwd_per_ep)
    ql_agent.env.close()
    ql_agent.plot_rwds(rwds)