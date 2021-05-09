"""
This is file includes the main function for gym atari reinforcement learning.
"""
import gym
import numpy as np

from argparse_decorate import init_parser, add_arg
from deep_q_learning import DQLAtari
from sac_discrete import SoftActorCriticsDiscrete


@init_parser()
@add_arg('--start_episode', type=int, default=8200, help='A number for start episode index.')
@add_arg('--eval', type=bool, default=False, help='True means evaluate model only.')
@add_arg('--game_index', type=int, default=1, choices=[0, 1], help='Represent Breakout and MsPacman respectively.')
@add_arg('--memory_size', type=int, default=65000, help='The size of the memory space.')
@add_arg('--start_epsilon', type=float, default=0.1, help='The probability for random actions.')
@add_arg('--reward_scale', type=int, default=3, help='The scale of each reward after steps.')
@add_arg('--agent', type=str, default='dql', choices=['dql', 'dsac'],
         help='Deep Q-learning and discrete soft Actor-Critics algorithms.')
def main(**kwargs):
    """
    The main function for gym atari reinforcement learning.
    """
    atari_game = ['Breakout-v0', 'MsPacman-v0']
    img_size = [(4, 105, 80), (4, 86, 80)]
    env = gym.make(atari_game[kwargs['game_index']])
    memory_par = (kwargs['memory_size'], img_size[kwargs['game_index']])
    action_space = np.array([i for i in range(env.action_space.n)], dtype=np.uint8)
    game = (atari_game[kwargs['game_index']], env)
    if kwargs['start_episode'] == 0:
        path = None
    else:
        path = './' + atari_game[kwargs['game_index']]

    if kwargs['agent'] == 'dql':
        agent = DQLAtari(memory_par=memory_par,
                         action_space=action_space,
                         game=game,
                         start_epsilon=kwargs['start_epsilon'])
    elif kwargs['agent'] == 'dsac':
        agent = SoftActorCriticsDiscrete(memory_par=memory_par,
                                         action_space=action_space,
                                         game=game,
                                         reward_scale=kwargs['reward_scale'])
    else:
        agent = None

    if agent is not None:
        agent.simulate(net_path=path, start_episodes=kwargs['start_episode'], eval=kwargs['eval'])


if __name__ == '__main__':
    main()
