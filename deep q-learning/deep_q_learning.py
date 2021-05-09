"""
An agent implement deep Q-learning algorithm.
"""
import torch
import numpy as np

from dql_net import DqlNet
from general_agent import AtariAgent


class DQLAtari(AtariAgent):
    def __init__(self, action_space, memory_par, game, start_epsilon):
        """
        This is an agent for deep Q-learning algorithm.

        Args:
            action_space: An array or list like, contains all of the actions of the environment.
            memory_par: A tuple, including the size of the memory space and multi-frames image size.
            game: A tuple, including the game name and the gym environment for this game.
            start_epsilon: A float represents the satrt epsilon.
        """
        super().__init__(action_space, memory_par, game)
        self.learn_replace = 1000
        self.step_num = 2
        self.epsilon = start_epsilon
        with torch.no_grad():
            self.target_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)
        self.behavior_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)

    def predict_action_q(self, state):
        """
        Calculate q values about different actions through certain state.
        """
        with torch.no_grad():
            return self.behavior_net.forward(state[None, ...].to(self.behavior_net.device))

    def get_action(self, s, eval=False):
        """
        Choose action under certain policy with epsilon-greedy method.

        Returns:
            An action chose by policy and epsilon-greedy method.
        """
        action_q = self.predict_action_q(s)
        a = torch.argmax(action_q).item()
        if np.random.rand() < self.epsilon:
            return self.action_space[np.random.randint(self.action_space_len)]
        else:
            return a

    def update_target(self):
        """
        Update the target network under certain learning times.
        """
        if self.learn_cur % self.learn_replace == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

    def update_episode(self):
        """
        Update the epsilon after each learning.
        """
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > 0.1 else 0.1

    def learn(self):
        """
        Learn from the memory and update the network.
        """
        if self.step_count < 1000:
            return

        s, a, r, s_, t = self.sample()
        self.update_target()
        q_behavior = self.behavior_net.forward(s.to(self.behavior_net.device).float())
        q_behavior_a = q_behavior[np.arange(self.batch_size), np.array(a)]
        q_target = self.target_net.forward(s_.to(self.target_net.device).float())
        q_target_max = torch.max(q_target, dim=1)[0]

        q_target_max[t] = 0.0
        q = r.to(self.behavior_net.device).float() + self.gamma * q_target_max

        self.behavior_net.optimizer.zero_grad()
        self.behavior_net.loss(q, q_behavior_a).backward()
        self.behavior_net.optimizer.step()
        self.learn_cur += 1

    def load_model(self, net_path, eval=False, start_episodes=0):
        """
        Load existent model and memory. If eval is True the only load behavior network.

        Args:
            net_path: The path that the model saved.
            eval: A bool, True represents evaluate RL model only.
            start_episodes: An integer represents the episodes num at the start time.
        """
        if net_path is not None:
            self.behavior_net.load_state_dict(
                torch.load(net_path + '/' + self.game_name + '{}.pth'.format(start_episodes)))
            if eval:
                self.behavior_net.eval()
            else:
                self.target_net.load_state_dict(torch.load(net_path + '/target{}.pth'.format(start_episodes)))
                self.behavior_net.optimizer.load_state_dict(
                    torch.load(net_path + '/optimizer{}.pth'.format(start_episodes)))
                self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))

    def process_results(self, episode):
        """
        Salve models and plot results after certain episodes.
        """
        if episode % 10 == 9:
            ave = np.mean(self.scores[episode - 9:episode])
            print('Episodes: {}, AveScores: {}, Epsilon: {}'.format(episode + 1, ave, self.epsilon))

        if episode % 200 == 199:
            s1 = './' + self.game_name + '/'
            s_pth = '{}.pth'.format(episode + 1)
            torch.save(self.behavior_net.state_dict(), s1 + self.game_name + s_pth)
            torch.save(self.target_net.state_dict(), s1 + 'target' + s_pth)
            torch.save(self.behavior_net.optimizer.state_dict(), s1 + 'optimizer' + s_pth)
            np.save(s1 + 'scores{}.npy'.format(episode + 1), self.scores)

            self.plot_array(episode)
            print('Model salved!')
            print('Total {} frames!'.format(self.frames_count))
