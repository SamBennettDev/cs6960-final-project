import chess
import chess.svg
import chess.pgn
import RandomBot
from ChessBot import ChessBot
from EvaluatePos import *
from Visualize import *

def trainBot(bot, num_games):
    board = chess.Board()
    chess_drawer = ChessBoardDrawer(600, 600, board)
    bot.save_model()
    for i in range(num_games):

        while not board.is_game_over():
            # count += 1
            # print(f"\n{count}]\n")
            if board.turn:  # whites turn
                bef_adv = calc_advantage(board)
                bot_move = bot.make_move(board)
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
            
        print(board.outcome())

sam_bot = ChessBot("sam")
trainBot(sam_bot, 20)
