import chess
import chess.svg
import chess.pgn
import MiniMax
import random
from EvaluatePos import *

count = 0
depth = 5
movehistory = []
game = chess.pgn.Game()
board = chess.Board()
while not board.is_game_over():
    if board.turn:
        count += 1
        print(f'\n{count}]\n')
        move = MiniMax.move(board, depth)
        board.push(move)
        print(board)
        print()
    else:
        board.push(random.choice(list(board.legal_moves)))
        print(board)
        
game.add_line(movehistory)
game.headers["Event"] = "Self Tournament 2020"
game.headers["Site"] = "Pune"
game.headers["Round"] = 1
game.headers["White"] = "Ai"
game.headers["Black"] = "Ai"
game.headers["Result"] = str(board.result())
print(game)
print(board.outcome())

