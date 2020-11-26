from model import Residual_CNN
from agent import Agent

import gym

import wandb
# wandb.init(project="Data Challenge 2 - Go")

# Hyperparameters
BOARD_SIZE = 9
REG_CONST = 0.0001
LEARNING_RATE = 0.1
KOMI = 0
CPUCT = 1
MCTS_SIMS = 50

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
current_NN = Residual_CNN(REG_CONST, LEARNING_RATE, env.observation_space.shape, BOARD_SIZE^2, HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(REG_CONST, LEARNING_RATE, env.observation_space.shape, BOARD_SIZE^2, HIDDEN_CNN_LAYERS)
best_NN.model.set_weights(current_NN.model.get_weights())


# Init agents
current_player = Agent('current_player', MCTS_SIMS, CPUCT, current_NN)
best_player = Agent('best_player', MCTS_SIMS, CPUCT, best_NN)