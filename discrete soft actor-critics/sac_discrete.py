"""
An agent implement Soft Actor-Critics algorithm for discrete action space.
"""
import torch
import numpy as np

from torch import optim
from general_net import CnnExtractor
from sacd_net import DenseNet, PolicyNetwork
from general_agent import AtariAgent


class SoftActorCriticsDiscrete(AtariAgent):
    def __init__(self, action_space, memory_par, game, reward_scale):
        """
        This is a Soft Actor-Critics agent for discrete action space.

        Args:
            action_space: An array or list like, contains all of the actions of the environment.
            memory_par: A tuple, including the size of the memory space and multi-frames image size.
            game: A tuple, including the game name and the gym environment for this game.
            reward_scale: The scale of each reward after steps.
        """
        super(SoftActorCriticsDiscrete, self).__init__(action_space, memory_par, game)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.soft = False
        self.learn_replace = 4000
        self.step_num = 3
        self.reward_scale = reward_scale
        self.conv_net = CnnExtractor(img_size=self.multi_frames_size)
        self.actor = PolicyNetwork(self.conv_net.flatten_size, output_num=self.action_space_len)
        self.critic1 = DenseNet(self.conv_net.flatten_size, output_num=self.action_space_len)
        self.critic2 = DenseNet(self.conv_net.flatten_size, output_num=self.action_space_len)
        with torch.no_grad():
            self.critic1_target = DenseNet(self.conv_net.flatten_size, output_num=self.action_space_len)
            self.critic2_target = DenseNet(self.conv_net.flatten_size, output_num=self.action_space_len)

        self.target_entropy = 0.98 * (-np.log(1 / self.action_space_len))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.critic1_optimiser = optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic2_optimiser = optim.Adam(self.critic2.parameters(), lr=0.0003)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.alpha_optimiser = optim.Adam([self.log_alpha], lr=0.0003)

    def cal_critic_loss(self, s, a, r, t, s_):
        """
        Calculate two critics' loss.
        """
        with torch.no_grad():
            _, a_probabilities, log_a_probabilities = \
                self.actor.sample_action(self.conv_net.conv(s_.to(self.device).float()))
            target_q1 = self.critic1_target(self.conv_net.conv(s_.to(self.device).float()))
            target_q2 = self.critic2_target(self.conv_net.conv(s_.to(self.device).float()))
            q_target = (a_probabilities * (torch.min(target_q1, target_q2) - self.alpha * log_a_probabilities)).sum(
                dim=1)
            terminal = (~t.to(self.device))
            q_ = r.to(self.device).float() + self.gamma * terminal * q_target

        q1 = self.critic1(self.conv_net.conv(s.to(self.device).float()))[np.arange(self.batch_size), np.array(a)]
        q2 = self.critic2(self.conv_net.conv(s.to(self.device).float()))[np.arange(self.batch_size), np.array(a)]
        q1_loss = torch.nn.functional.mse_loss(q1, q_)
        q2_loss = torch.nn.functional.mse_loss(q2, q_)
        return q1_loss, q2_loss

    def update_critics(self, s, a, r, t, s_):
        """
        Update two critics after certain steps.
        """
        q1_loss, q2_loss = self.cal_critic_loss(s, a, r, t, s_)

        self.critic1_optimiser.zero_grad()
        q1_loss.backward()
        self.critic1_optimiser.step()

        self.critic2_optimiser.zero_grad()
        q2_loss.backward()
        self.critic2_optimiser.step()

    def cal_actor_alpha_loss(self, s_):
        """
        Calculate loss for actor and alpha.
        """
        _, a_probabilities, log_a_probabilities = self.actor.sample_action(
            self.conv_net.conv(s_.to(self.device).float()))
        with torch.no_grad():
            q1 = self.critic1(self.conv_net.conv(s_.to(self.device).float()))
            q2 = self.critic2(self.conv_net.conv(s_.to(self.device).float()))
            q = torch.min(q1, q2)

        actor_loss = (a_probabilities * (self.alpha * log_a_probabilities - q)).sum(dim=1).mean()
        probabilities = (a_probabilities * log_a_probabilities).sum(dim=1)
        alpha_loss = -(self.log_alpha * (probabilities + self.target_entropy).detach()).mean()
        return actor_loss, alpha_loss

    def update_actor_alpha(self, s_):
        """
        Update actor and alpha after certain steps.
        """
        actor_loss, alpha_loss = self.cal_actor_alpha_loss(s_)

        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.alpha = self.log_alpha.exp()

    def update_target(self):
        """
        Update or replace target critics.
        """
        if self.soft:
            self.sotf_update_target(self.critic1_target, self.critic1)
            self.sotf_update_target(self.critic2_target, self.critic2)
        else:
            if self.learn_cur % self.learn_replace == 0:
                self.critic1_target.load_state_dict(self.critic1.state_dict())
                self.critic2_target.load_state_dict(self.critic2.state_dict())

    def learn(self):
        """
        All of the models learn from the memory after certain steps.
        """
        if self.step_count < 1000 or self.step_count % 4 != 0:
            return
        s, a, r, s_, t = self.sample()
        self.update_critics(s, a, r, t, s_)
        self.update_actor_alpha(s_)
        self.update_target()
        self.learn_cur += 1

    def get_action(self, s, eval=False):
        """
        Get action in three different ways. If eval symbol is True, we choose action according the biggest actions'
        value. Otherwise, we choose actions randomly at first and then sample from actions' probabilities.
        """
        if eval:
            with torch.no_grad():
                conv_s = self.conv_net.conv(s[None, ...].to(self.device))
                action = self.actor.get_best_action(conv_s)
        else:
            if self.frames_count < 20000:
                action = self.action_space[np.random.randint(self.action_space_len)]
            else:
                with torch.no_grad():
                    conv_s = self.conv_net.conv(s[None, ...].to(self.device))
                    action, _, _ = self.actor.sample_action(conv_s)
        return action.item()

    def save_memory(self, r, action, multi_frames_, done):
        """
        Save episodes experience into memory space.
        """
        if r > 0 or (r <= 0 and np.random.rand() < 0.1):
            if self.reward_scale != 0:
                r = min(r * self.reward_scale, self.reward_scale)
            self.memory.store_sars_(self.multi_frames.to('cpu'),
                                    torch.Tensor([action]), torch.Tensor([r]), multi_frames_, torch.Tensor([done]))

    def process_results(self, episode):
        """
        Salve models and plot results after certain episodes.
        """
        if episode % 10 == 9:
            ave = np.mean(self.scores[episode - 9:episode])
            print('Episodes: {}, AveScores: {}, Alpha: {}'.format(episode + 1, ave, self.alpha.item()))

        if episode % 200 == 199:
            self.save_episode_models(episode)
            self.plot_array(episode)
            print('Model salved!')
            print('Total {} frames!'.format(self.frames_count))

    def save_episode_models(self, episode):
        """
        Save models and scores in certain episodes.
        """
        s1 = './' + self.game_name + '/'
        s_pth = '{}.pth'.format(episode + 1)
        torch.save(self.actor.state_dict(), s1 + 'actor' + s_pth)
        torch.save(self.critic1.state_dict(), s1 + 'critic1_' + s_pth)
        torch.save(self.critic2.state_dict(), s1 + 'critic2_' + s_pth)
        torch.save(self.critic1_target.state_dict(), s1 + 'critic1_target' + s_pth)
        torch.save(self.critic2_target.state_dict(), s1 + 'critic2_target' + s_pth)

        torch.save(self.log_alpha, s1 + 'log_alpha' + s_pth)

        torch.save(self.actor_optimiser.state_dict(), s1 + 'actor_optimizer' + s_pth)
        torch.save(self.critic1_optimiser.state_dict(), s1 + 'critic1_optimizer' + s_pth)
        torch.save(self.critic2_optimiser.state_dict(), s1 + 'critic2_optimizer' + s_pth)
        torch.save(self.alpha_optimiser.state_dict(), s1 + 'alpha_optimizer' + s_pth)
        np.save(s1 + 'scores{}.npy'.format(episode + 1), self.scores)

    def load_model(self, net_path, eval=False, start_episodes=0):
        """
        Load existent model and memory. If eval is True the only load behavior network.

        Args:
            net_path: The path that the model saved.
            eval: A bool, True represents evaluate RL model only.
            start_episodes: An integer represents the episodes num at the start time.
        """
        if net_path is not None:
            self.actor.load_state_dict(torch.load(net_path + '/actor{}.pth'.format(start_episodes)))
            if eval:
                self.actor.eval()
            else:
                s1 = net_path + '/'
                s2 = '{}.pth'.format(start_episodes)
                self.critic1.load_state_dict(torch.load(s1 + 'critic1_' + s2))
                self.critic2.load_state_dict(torch.load(s1 + 'critic2_' + s2))
                self.critic1_target.load_state_dict(torch.load(s1 + 'critic1_target' + s2))
                self.critic2_target.load_state_dict(torch.load(s1 + 'critic2_target' + s2))
                self.log_alpha = torch.load(s1 + 'log_alpha' + s2, map_location=self.device)

                self.actor_optimiser.load_state_dict(torch.load(s1 + 'actor_optimizer' + s2))
                self.critic1_optimiser.load_state_dict(torch.load(s1 + 'critic1_optimizer' + s2))
                self.critic2_optimiser.load_state_dict(torch.load(s1 + 'critic2_optimizer' + s2))
                self.alpha_optimiser.load_state_dict(torch.load(s1 + 'alpha_optimizer' + s2))

                self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))
