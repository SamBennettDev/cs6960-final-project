import chess
import chess.svg
import chess.pgn
import RandomBot
from ChessBot import ChessBot
from EvaluatePos import *
from Visualize import *

def runBot(bot, num_games):
    botWins = 0
    randWins = 0
    draws = 0
    reasonDraw = []

    for i in range(num_games):
        board = chess.Board()
        #chess_drawer = ChessBoardDrawer(600, 600, board)
        while not board.is_game_over():
            if board.turn:  # whites turn
                bot_move = bot.make_move(board)
                board.push(bot_move)
                #chess_drawer.update_display()

            else:  # blacks turn
                board.push(RandomBot.move(board))
                #chess_drawer.update_display()
            
        outcome = board.outcome()
        if outcome.winner == chess.WHITE:
            botWins+=1
        elif outcome.winner == chess.BLACK:
            randWins+=1
        else:
            draws+=1
            reasonDraw.append(outcome.termination)

        print("Game #" + str(i) + ": " + str(outcome))
        
    print("%s won %i times, lost %i times, and tied %i times." % (bot.trainer_name, botWins,randWins,draws))
    print(reasonDraw)



untrained = ChessBot("stockfish")
runBot(untrained, 20)