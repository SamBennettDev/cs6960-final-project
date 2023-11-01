import random


def move(board):
    return random.choice(list(board.legal_moves))
