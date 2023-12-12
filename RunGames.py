import chess
import chess.svg
import chess.pgn
import RandomBot
import os
from random import sample
from ChessBot import ChessBot


def randomRuns(num_games=100):
    currentdirect = os.getcwd()
    dir_list = os.listdir(currentdirect + "//models")
    num_bots = len(dir_list)

    results = {}

    for i in range(num_bots):
        name = dir_list[i]
        name = name[0:len(name)-4]
        results[name] = []
    
    for i in range(num_bots):
        name1 = dir_list[i]
        name1 = name1[0:len(name1)-4]
        bot1 = ChessBot(name1)
        
        for j in range(i+1,num_bots):
            name2 = dir_list[j]
            name2 = name2[0:len(name2)-4]
            bot2 = ChessBot(name2)

            for game in range(num_games):
                outcome = runGame(bot1, bot2)
                if outcome.winner == chess.WHITE:
                    results[name1].append(1)
                    results[name2].append(-1)
                elif outcome.winner == chess.BLACK:
                    results[name1].append(-1)
                    results[name2].append(1)
                else:
                    results[name1].append(0)
                    results[name2].append(0)

                print(name1 + " vs " + name2 + " Game #" + str(game) + ": " + str(outcome))
        
    for bot in results:
        result = results[bot]
        print("%s won %i times, lost %i times, and tied %i times." % (bot, result.count(1),result.count(-1),result.count(0)))
    return

def runGame(bot1, bot2=None):
    board = chess.Board()
    #chess_drawer = ChessBoardDrawer(600, 600, board)
    while not board.is_game_over():
        if board.turn:  # whites turn
            bot_move = bot1.make_move(board)
            board.push(bot_move)
            #chess_drawer.update_display()

        else:  # blacks turn
            if bot2 is None:
                board.push(RandomBot.move(board))
            else:
                bot_move = bot2.make_move(board)
                board.push(bot_move)
            #chess_drawer.update_display()
    return board.outcome()

def runBot(bot1, bot2=None, num_games=100):
    whiteWins = 0
    blackWins = 0
    draws = 0
    reasonDraw = []

    for i in range(num_games):   
        outcome = runGame(bot1, bot2)
        if outcome.winner == chess.WHITE:
            whiteWins+=1
        elif outcome.winner == chess.BLACK:
            blackWins+=1
        else:
            draws+=1
            reasonDraw.append(outcome.termination)

        print("Game #" + str(i) + ": " + str(outcome))
        
    print("%s won %i times, lost %i times, and tied %i times." % (bot1.trainer_name, whiteWins,blackWins,draws))
    if bot2 is not None:
        print("%s won %i times, lost %i times, and tied %i times." % (bot2.trainer_name, blackWins,whiteWins,draws))
    print(reasonDraw)




def main():
    while True:
        typeRun = input("Would you like to: \n\t1. Run with two specified bots\n\t2. Run a specified bot against random\n\t3. Run with all possible bots against each other\n\t0. Exit\nPlease enter the number: ")

        if typeRun == "1":
            currentdirect = os.getcwd()
            dir_list = os.listdir(currentdirect + "//models")
            text = "What two bots would you like to run against each other?\n"
            for i in range(len(dir_list)):
                text += "\t" + str(i)+ ". " + str(dir_list[i]) + "\n"
            text += "Enter the two number with a space between: "
            choice = input(text)

            # get choices and put them in bots
            choice = choice.split(" ")
            choice1 = int(choice[0])
            choice2 = int(choice[1])
            filename1 = dir_list[choice1]
            filename2 = dir_list[choice2]
            bot1 = ChessBot(filename1[0:len(filename1)-4])
            bot2 = ChessBot(filename2[0:len(filename2)-4])

            numruns = input("How many runs would you like to do? ")

            runBot(bot1, bot2, int(numruns))

        elif typeRun == "2":
            # Initialize your chess bot
            currentdirect = os.getcwd()
            dir_list = os.listdir(currentdirect + "//models")
            text = "What bot would you like to run against random?\n"
            for i in range(len(dir_list)):
                text += "\t" + str(i)+ ". " + str(dir_list[i]) + "\n"
            text += "Enter the number: "
            choice = input(text)
            numruns = input("How many runs would you like to do? ")
            filename = dir_list[int(choice)]
            bot = ChessBot(filename[0:len(filename)-4])
            runBot(bot, num_games=int(numruns))
        elif typeRun == "3":
            numruns = input("How many runs per matchup would you like to do? ")
            randomRuns(int(numruns))
        else:
            break



if __name__ == "__main__":
    main()