import chess
import chess.svg
import chess.pgn
import RandomBot
from EvaluatePos import *
from Visualize import *

count = 0
depth = 5
movehistory = []
game = chess.pgn.Game()
board = chess.Board()
chess_drawer = ChessBoardDrawer(600, 600, board)

while not board.is_game_over():
    count += 1
    print(f"\n{count}]\n")
    if board.turn:
        board.push(RandomBot.move(board))
        chess_drawer.update_display()
        print(board)
        print()
    else:
        board.push(RandomBot.move(board))
        chess_drawer.update_display()
        print(board)
    print(calc_advantage(board))

game.add_line(movehistory)
game.headers["Event"] = "Self Tournament 2020"
game.headers["Site"] = "Pune"
game.headers["Round"] = 1
game.headers["White"] = "Ai"
game.headers["Black"] = "Ai"
game.headers["Result"] = str(board.result())
print(game)
print(board.outcome())
