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
    # count += 1
    # print(f"\n{count}]\n")
    if board.turn:  # whites turn
        bef_adv = calc_advantage(board)
        bot_move = sam_bot.make_move(board)
        board.push(bot_move)
        after_adv = -calc_advantage(board)
        adv_change = after_adv - bef_adv
        chess_drawer.update_display()
        # print(board)
        print("advantage change " + str(adv_change))
        board.pop()
        trainer_move = chess.Move.from_uci(
                input(
                    "What's a better move? ["
                    + str([board.uci(move) for move in board.legal_moves])
                    + "]    "
                )
            )
        board.push(trainer_move)
        trainer_after_adv = -calc_advantage(board)
        trainer_adv_change = trainer_after_adv - bef_adv
        reward = adv_change - trainer_adv_change
        print("reward: " + str(reward))

        chess_drawer.update_display()

        # print(board)
        # print()
    else:  # blacks turn
        board.push(RandomBot.move(board))
        chess_drawer.update_display()
        # print(board)
    

game.add_line(movehistory)
game.headers["Event"] = "Self Tournament 2020"
game.headers["Site"] = "Pune"
game.headers["Round"] = 1
game.headers["White"] = "Ai"
game.headers["Black"] = "Ai"
game.headers["Result"] = str(board.result())
print(game)
print(board.outcome())
