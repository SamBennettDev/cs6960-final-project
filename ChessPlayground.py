import chess
import chess.svg
import chess.pgn
import RandomBot
from ChessBot import ChessBot
from ChessBotModel import ChessBotModel
from EvaluatePos import *
from Visualize import *

count = 0
movehistory = []
game = chess.pgn.Game()
board = chess.Board()
chess_drawer = ChessBoardDrawer(600, 600, board)
model = ChessBotModel()
sam_bot = ChessBot(model, "Sam")

while not board.is_game_over():
    count += 1
    print(f"\n{count}]\n")
    if board.turn:
        board.push(RandomBot.move(board))
        chess_drawer.update_display()
        print(board)
        print()
    else:
        board.push(sam_bot.make_move(board))
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
