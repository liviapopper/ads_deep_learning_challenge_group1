from gobang.Game import GobangGame
from NNet import NNetWrapper
from Coach import Coach
from utils import dotdict

import datetime

import wandb
wandb.init(project="AlphaZero", entity="ineedsugar", name="lobotomize1", config={"NeuralNetworkSetup": "standard"})

now = datetime.datetime.now()
args = dotdict({
    'numIters': 3,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': f'./temp/2020-12-03 16-52-23',
    'load_model': False,
    'load_folder_file': ('./temp/2020-12-03 16-52-23','temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    # Create a 9x9 board
    game = GobangGame(n=9, nir=9)

    network = NNetWrapper(game)
    if args.load_model:
        network.load_checkpoint(args.checkpoint)
    
    coach = Coach(game, network, args)
    coach.learn()

if __name__ == "__main__":
    main()