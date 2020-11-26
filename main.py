from model import Residual_CNN

import gym

import wandb
wandb.init(project="Data Challenge 2 - Go")

# Hyperparameter
REG_CONST = 0.0001
LEARNING_RATE = 0.1
BOARD_SIZE = 9
KOMI = 0

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)},
    {'filters':75, 'kernel_size': (4,4)},
    {'filters':75, 'kernel_size': (4,4)},
    {'filters':75, 'kernel_size': (4,4)},
    {'filters':75, 'kernel_size': (4,4)},
    {'filters':75, 'kernel_size': (4,4)}
]


# Init environment
env = gym.make('gym_go:go-v0', size=BOARD_SIZE, komi=KOMI, reward_method='real')

# Init models
current_NN = Residual_CNN(REG_CONST, LEARNING_RATE, env.observation_space.shape, env.action_space.shape[0], HIDDEN_CNN_LAYERS)
