"""
A general gym-atari agent that can be combined with special algorithm.
"""
import torch
import cv2
import numpy as np
import memory_space as ms
import matplotlib.pyplot as plt


class AtariAgent:
    def __init__(self, action_space: np.array, memory_par: tuple, game: tuple, reward_clip: bool):
        """
        This is an general gym-atari agent.
        In this class, some common RL parameters and the whole RL frame have been defined.
        For special algorithm agent, initialization and some functions should be redefined.
        The functions are as follow: load_model, get_action, learn and process_results.
        Other functions also could be redefined for special requirements.

        Args:
            action_space: An array contains all actions.
            memory_par: Including the size of the memory space and multi-frames image size.
            game: Including the game name and the gym environment.
            reward_clip: Clip reward in [-1, 1] range if True.
        """
        self.game_name, self.environment, self.live_penalty_mode = game
        self.reward_clip = reward_clip
        self.gamma = 0.99
        self.epsilon, self.epsilon_decay, self.mini_epsilon, self.final_epsilon = 0.1, 5e-6, 0.05, 0.01
        self.learn_start_step = 20000
        self.learn_cur, self.learn_replace, self.learn_interval = 0, 1000, 4
        self.no_op_num = 7
        self.episodes = 100000
        self.explore_frame = 5e6
        self.action_space, self.action_space_len = action_space, len(action_space)
        self.frames_num = memory_par[1][0]
        self.multi_frames_size, self.single_img_size = memory_par[1], (1, *memory_par[1][1:])
        self.memory = ms.Memory(*memory_par)
        self.multi_frames = torch.zeros(size=memory_par[1], dtype=torch.float32)
        self.scores = np.zeros(self.episodes, dtype=np.float16)
        self.batch_size = 32
        self.frames_count = 0
        self.step_num, self.step_count = 4, 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.live_penalty_mode:
            self.max_lives, self.live_penalty, self.cur_lives = self._init_live_penalty_mode()

    def _init_live_penalty_mode(self):
        _ = self.environment.reset()
        _, _, _, info = self.environment.step(0)
        return info['ale.lives'], False, info['ale.lives']

    def preprocess_observation(self, observation: np.array) -> torch.Tensor:
        """
        Transform rgb observation image to a smaller gray image.

        Args:
            observation: The image data.

        Returns:
            tensor_observation: A sequence represents a gray image.
        """
        image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self.single_img_size[1:], interpolation=cv2.INTER_AREA)
        tensor_observation = torch.from_numpy(image)
        return tensor_observation

    def reset_episode(self, environment):
        """
        Rest the environment at the begging of a episode.

        Args:
            environment: The gym atari game environment.

        Returns:
            The bool represents whether the episode has done, 0 represents the score.
        """
        observation = self.preprocess_observation(environment.reset())
        if self.live_penalty_mode:
            self.cur_lives = self.max_lives
            self.live_penalty = False
        for i in range(self.frames_num):
            self.multi_frames[i] = observation
        multi_frames_ = self.multi_frames
        done, rewards = False, 0
        for _ in range(self.no_op_num):
            multi_frames_, step_rewards, done = self.go_steps(multi_frames_, 0)
            rewards += step_rewards
        self.multi_frames = multi_frames_
        return done, rewards, multi_frames_

    def update_multi_frames(self, observation_: torch.Tensor) -> torch.Tensor:
        """
        Add new observation into the multi frames space and delete the oldest one.

        Args:
            observation_: The new observation getting through current action.

        Returns:
            multi_frames_: The new multi frames space.
        """
        multi_frames_ = self.multi_frames.clone().detach()
        for i in range(self.frames_num - 1):
            multi_frames_[self.frames_num - 1 - i, :] = multi_frames_[self.frames_num - 2 - i, :]
        multi_frames_[0, :] = observation_
        return multi_frames_

    def sample(self):
        """
        Sample batch size memory(sars_t) from memory space.
        """
        return self.memory.sample(self.batch_size)

    def load_model(self, net_path: str, eval: bool, start_episodes: int):
        """
        Load saved model according different algorithm.

        Args:
            net_path: The path that contains all of models.
            eval: True represents evaluate only.
            start_episodes: The num of the start episode.
        """
        pass

    def get_action(self, s: torch.Tensor, eval=False) -> int:
        """
        Get action through special algorithm.

        Args:
            eval: True represents evaluate mode.
            s: The sequence contains the state of the environment.

        Returns:
            The action generated bt the algorithm under certain state.
        """
        pass

    def learn(self):
        """
        Update the whole algorithm under certain cases.
        """
        pass

    def soft_update_target(self, target_model, behavior_model):
        """
        Update target network softly.
        """
        for target_param, local_param in zip(target_model.parameters(), behavior_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_memory(self, r, action, multi_frames_, done):
        """
        Save episodes experience into memory space.
        """
        if self.reward_clip:
            r = min(max(r, -self.step_num), self.step_num)
        if self.live_penalty_mode and self.live_penalty:
            r = -1
            self.live_penalty = False
        self.memory.store_sars_(self.multi_frames.to('cpu'),
                                torch.Tensor([action]), torch.Tensor([r]), multi_frames_, torch.Tensor([done]))

    def go_steps(self, multi_frames_: torch.Tensor, action: int):
        """
        Go several steps under a certain action.
        """
        step_rewards, done = 0, None
        for _ in range(self.step_num):
            observation_, reward, done, info = self.environment.step(action)
            if self.live_penalty_mode:
                if info['ale.lives'] != self.cur_lives:
                    self.live_penalty = True
                    self.cur_lives = info['ale.lives']
            step_rewards += reward
            multi_frames_ = self.update_multi_frames(self.preprocess_observation(observation_))
        self.step_count += 1
        return multi_frames_, step_rewards, done

    def update_episode(self):
        """
        Update some parameters after an episode in some algorithms.
        """
        pass

    def simulate(self, net_path=None, start_episodes=0, eval=False, start_frames=0):
        """
        This is the general RL frame, including the whole process.
        Through 'eval' parameter, we can switch mode between training and evaluation.

        Args:
            net_path: The path include model or data files.
            start_episodes: The num represents the start episode, using in refresher training.
            eval: True represents evaluate only.
            start_frames: The num of the start frames.
        """
        self.frames_count = start_frames
        self.load_model(net_path, eval, start_episodes)
        for episode in range(start_episodes, self.episodes):
            done, score, multi_frames_ = self.reset_episode(self.environment)
            while not done:
                self.environment.render()
                action = self.get_action(self.multi_frames, eval)
                multi_frames_, step_rewards, done = self.go_steps(multi_frames_, action)
                score += step_rewards
                self.frames_count += self.step_num
                if not eval:
                    self.save_memory(step_rewards, action, multi_frames_, done)
                    self.learn()

                self.multi_frames = multi_frames_
                self.update_episode()
            self.scores[episode] = score
            self.process_results(episode, eval)

    def process_results(self, episode: int, eval=False):
        """
        Process result in certain episodes, saving model, plotting results and so on.
        """
        pass

    def plot_array(self, episode: int):
        """
        Plot moving window averages and scores.
        """
        N = 100
        result = np.convolve(self.scores[0:episode + 1], np.ones((N,)) / N, mode='valid')
        plt.figure()
        plt.plot(result)
        plt.xlabel('Game Times')
        plt.ylabel('100 episodes moving window average')
        plt.show()

        plt.plot(self.scores[0:episode + 1])
        plt.xlabel('Game Times')
        plt.ylabel('Scores')
        plt.show()
