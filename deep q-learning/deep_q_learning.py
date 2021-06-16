"""
An agent implement Deep Q-Learning algorithm.
"""
import torch
import numpy as np

from dql_net import DqlNet
from general_agent import AtariAgent


class DQLAtari(AtariAgent):
    def __init__(self, action_space: np.array, memory_par: tuple, game: tuple, epsilon: tuple, reward_clip: bool):
        """
        This is an agent for Deep Q-Learning algorithm.

        Args:
            action_space: An array contains all actions.
            memory_par: Including the size of the memory space and multi-frames image size.
            game: Including the game name and the gym environment.
            epsilon: Includes the epsilon and minimum epsilon.
            reward_clip: Clip reward in [-1, 1] range if True.
        """
        super().__init__(action_space, memory_par, game, reward_clip)
        self.episodes = 100000
        self.learn_replace = 5000
        self.epsilon, self.mini_epsilon = epsilon
        with torch.no_grad():
            self.target_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)
        self.behavior_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)

    def predict_action_q(self, state):
        """
        Calculate q values about different actions under certain state.
        """
        with torch.no_grad():
            return self.behavior_net.forward(state[None, ...].to(self.behavior_net.device))

    def get_action(self, s: torch.Tensor, eval=False) -> int:
        """
        Choose action under certain policy with epsilon-greedy method.

        Returns:
            An action for current state under certain policy.
        """
        action_q = self.predict_action_q(s)
        a = torch.argmax(action_q).item()
        if np.random.rand() < self.epsilon and not eval:
            return self.action_space[np.random.randint(self.action_space_len)]
        else:
            return a

    def update_target(self):
        """
        Update the target network under certain learning times.
        """
        if self.step_count % self.learn_replace == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

    def update_episode(self):
        """
        Update the epsilon after each learning.
        """
        if self.frames_count > self.explore_frame:
            self.epsilon = self.final_epsilon
        else:
            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.mini_epsilon else self.mini_epsilon

    def learn(self):
        """
        Learn from memory and update the network.
        """
        if self.step_count < self.learn_start_step or self.step_count % self.learn_interval != 0:
            return

        s, a, r, s_, t = self.sample()
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
        self.update_target()

    def load_model(self, net_path: str, eval=False, start_episodes=0):
        """
        Load existent model. If in evaluate mode then load behavior network only.

        Args:
            net_path: The path that contains all of models.
            eval: True represents evaluate only.
            start_episodes: The num of the start episode.
        """
        if eval:
            self.behavior_net.load_state_dict(
                torch.load(net_path + '/' + self.game_name + '.pth'))
            self.behavior_net.eval()
        if start_episodes != 0 and not eval:
            self.behavior_net.load_state_dict(
                torch.load(net_path + '/' + self.game_name + '{}.pth'.format(start_episodes)))
            self.target_net.load_state_dict(torch.load(net_path + '/target{}.pth'.format(start_episodes)))
            self.behavior_net.optimizer.load_state_dict(
                torch.load(net_path + '/optimizer{}.pth'.format(start_episodes)))
            self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))
            self.learn_cur += 1

    def process_results(self, episode, eval):
        """
        Salve models and plot results after certain episodes.
        """
        if episode % 10 == 9:
            ave = np.mean(self.scores[episode - 9:episode])
            print('Episodes: {}, AveScores: {}, Epsilon: {}, Steps: {}'.format(
                episode + 1, ave, self.epsilon, self.step_count))
        if eval:
            if episode % 100 == 99:
                s1 = './' + self.game_name + '/'
                np.save(s1 + 'scores_eval{}.npy'.format(episode + 1), self.scores)
                print('Evaluation results saved!')
        else:
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
