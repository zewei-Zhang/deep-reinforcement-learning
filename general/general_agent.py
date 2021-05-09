"""
A general gym-atari agent which could combine with special algorithm.
"""
import torch
import numpy as np
import memory_space as ms
import matplotlib.pyplot as plt

from general_net import generate_transform


class AtariAgent:
    def __init__(self, action_space, memory_par, game):
        """
        This is an general gym-atari agent.
        In this class, some common RL parameters and the whole RL frame have been defined.
        For special algorithm agent, initialization and some functions should be redefined.
        The functions are as follow: load_model, get_action, learn and process_results.
        Other functions also could be redefined for special requirements.

        Args:
            action_space: A list or list like, contains all of the possible actions.
            memory_par: A tuple, including the size of the memory space and multi-frames image size.
            game: A tuple, including the game name and the gym environment for this game.
        """
        self.game_name, self.environment = game
        self.gamma, self.epsilon, self.epsilon_decay, self.mini_epsilon = 0.99, 0.1, 1e-5, 0.1
        self.tau = 0.005
        self.step_num, self.step_count = 3, 0
        self.action_space, self.action_space_len = action_space, len(action_space)
        self.frames_num = memory_par[1][0]
        self.multi_frames_size, self.single_img_size = memory_par[1], (1, *memory_par[1][1:])
        self.memory = ms.Memory(*memory_par)
        self.transform = generate_transform()
        self.episodes = 50000
        self.multi_frames = torch.zeros(size=memory_par[1], dtype=torch.float32)
        self.scores = np.zeros(self.episodes, dtype=np.float16)
        self.batch_size = 64
        self.learn_cur, self.learn_replace = 0, 4000
        self.frames_count = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reset_episode(self, environment):
        """
        Rest the environment at the begging of a episode.

        Args:
            environment: The environment which need to be reset. Here is the gym atari environment.

        Returns:
            The bool represents whether the episode has done, 0 represents the score.
        """
        observation = self.preprocess_observation(environment.reset())
        for i in range(self.frames_num):
            self.multi_frames[i] = observation
        return False, 0, None

    def preprocess_observation(self, observation):
        """
        Preprocess observation getting from environment.

        Args:
            observation: The observation from environment, here is a image.

        Returns:
            tensor_observation: A tensor transformed from preprocessed observation.
        """
        if self.game_name == 'MsPacman-v0':
            observation = observation[1:172:2, ::2]
        else:
            observation = observation[::2, ::2]
        tensor_observation = self.transform(observation).reshape(self.single_img_size)
        return tensor_observation

    def update_multi_frames(self, observation_):
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

    def load_model(self, net_path, eval, start_episodes):
        """
        Load saved model according different algorithm.

        Args:
            net_path: The path that contains all of the models.
            eval: A bool, True represents evaluate only.
            start_episodes: The num of the episodes.
        """
        pass

    def get_action(self, s: np.array, eval=False):
        """
        Get action through special algorithm.

        Args:
            eval: A bool, True represents evaluate only.
            s: An array contains the state of the environment.

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
        if r > 0 or (r <= 0 and np.random.rand() < 0.1):
            self.memory.store_sars_(self.multi_frames.to('cpu'),
                                    torch.Tensor([action]), torch.Tensor([r]), multi_frames_, torch.Tensor([done]))

    def go_steps(self, multi_frames_, action):
        """
        Go several steps under a certain action.
        """
        if self.step_num == 1:
            rand_frames = 1
        else:
            rand_frames = np.random.randint(self.step_num - 1, self.step_num + 1)
        step_rewards, done = 0, None
        for _ in range(rand_frames):
            observation_, reward, done, _ = self.environment.step(action)
            step_rewards += reward
            multi_frames_ = self.update_multi_frames(self.preprocess_observation(observation_))
        self.step_count += 1
        return multi_frames_, step_rewards, done, rand_frames

    def update_episode(self):
        """
        Update some parameters after an episode in some algorithms.
        """
        pass

    def simulate(self, net_path=None, start_episodes=0, eval=False, start_frames=0):
        """
        This is the general RL frame, including the whole process.
        Through 'eval' parameter, we can switch training or evaluation mode.

        Args:
            net_path: The path include model or data files.
            start_episodes: The num represents the start episode, using in refresher training.
            eval: A bool, True represents evaluate only.
            start_frames: The num of the start frames.
        """
        self.frames_count = start_frames
        self.load_model(net_path, eval, start_episodes)
        for episode in range(start_episodes, self.episodes):
            done, score, multi_frames_ = self.reset_episode(self.environment)
            while not done:
                self.environment.render()
                action = self.get_action(self.multi_frames, eval)
                multi_frames_, step_rewards, done, rand_frames = self.go_steps(multi_frames_, action)
                score += step_rewards
                self.frames_count += rand_frames
                if not eval:
                    self.save_memory(step_rewards, action, multi_frames_, done)
                    self.learn()

                self.multi_frames = multi_frames_
                self.update_episode()
            self.scores[episode] = score
            self.process_results(episode)

    def process_results(self, episode):
        """
        Process result in certain episodes, including save model, plot results and so on.
        """
        pass

    def plot_array(self, episode):
        """
        Plot results in certain episodes.
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

    def record_video(self):
        """
        Record game in video form.
        """
        gym.wrappers.Monitor(self.environment, "./videos", force=True)
