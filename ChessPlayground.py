import chess
import chess.svg
import chess.pgn
import RandomBot
from ChessBot import ChessBot
from EvaluatePos import *
from Visualize import *
from stockfish import Stockfish

def trainBot(bot, num_games, human=True):
    board = chess.Board()
    chess_drawer = ChessBoardDrawer(600, 600, board)
    bot.save_model()
    for i in range(num_games):

        while not board.is_game_over():
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

                if human:
                    trainer_move = chess.Move.from_uci(
                            input(
                                "Bot made move " 
                                + str(bot_move)
                                + "\nWhat's a better move? ["
                                + str([board.uci(move) for move in board.legal_moves])
                                + "]    "
                            )
                        )
                else:
                    stockfish.set_fen_position(board.fen())
                    trainer_move = stockfish.get_best_move()

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

stockfish = Stockfish(path='stockfish/')
sam_bot = ChessBot("sam")
stockfish_bot = ChessBot("stockfish")
#trainBot(sam_bot, 10) #train by human
trainBot(stockfish_bot, 1000, False) # train by stockfish
