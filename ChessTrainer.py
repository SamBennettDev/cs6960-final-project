import chess
import chess.svg
import numpy as np
from ChessBot import ChessBot, from_move, to_move, to_np  # Assuming you have this module for the chess bot
import pygame
import sys
from stockfish import Stockfish
from EvaluatePos import calc_advantage

# Add other imports as needed

# ... (Your existing code)

square_size = 75  # Adjust the size based on your preference


def who(turn: bool):
    # Luna Chess
    """Who is playing, 1 for white -1 for black"""
    # return int(turn)
    return 1 if turn else -1


def get_canonical_form(board: chess.Board, player: chess.Color):
    # Luna chess
    """return state if player==1, else return -state if player==-1"""
    assert (who(board.turn) == player)

    if board.turn:
        return board
    else:
        return board.mirror()


def get_valid_moves(board: chess.Board, player: chess.Color):
    """Fixed size binary vector"""
    assert (who(board.turn) == player)

    acts = [0] * 4096
    for move in board.legal_moves:
        acts[from_move(move)] = 1

    return np.array(acts)


def get_symmetries(board, pi):
    return [(board, pi)]


def get_game_ended(board: chess.Board, player: chess.Color) -> float:
    """return 0 if not ended, 1 if player 1 won, -1 if player 1 lost"""

    outcome = board.outcome()
    reward = 0.0
    if outcome is not None:
        if outcome.winner is None:
            # draw, very little negative reward value
            reward = 1e-4
        else:
            if outcome.winner == board.turn:
                reward = 1.0
            else:
                reward = -1.0

    return reward


def get_action_prob(canonical_board: chess.Board, valid_moves, move) -> list:
    probs = [0] * 4096
    move_index = from_move(move)
    move_strength = 1
    probs = [(1 - move_strength) / (np.count_nonzero(valid_moves == 1) - 1)
             if valid_moves[a] == 1 else 0 for a in range(4096)]
    probs[move_index] = move_strength

    return probs


def mirror_move(move: chess.Move):
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))


def get_next_state(board: chess.Board, player: chess.Color, action):
    """Get next state given board and action"""
    # if player takes action on board, return next (board,player)
    # action must be a valid move

    assert (who(board.turn) == player)
    move = to_move(action)
    # if not board.turn:
    #     # assume the move comes from the canonical board...
    #     move = mirror_move(move)
    if move not in board.legal_moves:
        # could be a pawn promotion, which has an extra letter in UCI format
        move = chess.Move.from_uci(move.uci() + 'q')  # assume promotion to queen
        if move not in board.legal_moves:
            assert False, "%s not in %s" % (str(move), str(list(board.legal_moves)))
    board = board.copy()
    board.push(move)
    return board, who(board.turn)


def draw_board(screen, board, selected_square, legal_moves, bot_squares):
    # This function will draw the chessboard with highlighted pieces and legal moves

    colors = [(210, 210, 210), (30, 30, 30)]  # Define colors for light and dark squares

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            color = colors[(col + row) % 2]
            pygame.draw.rect(screen, color, square_to_rect(square, board.turn))

            piece = board.piece_at(square)
            if piece is not None:
                scale_factor = 0.8
                piece_name = chess.piece_name(piece.piece_type).lower()
                piece_image = pygame.image.load(
                    f"images/{'black' if piece.color == chess.BLACK else 'white'}-{piece_name}.png")
                piece_image = pygame.transform.scale(piece_image,
                                                     (int(square_size * scale_factor), int(square_size * scale_factor)))

                piece_rect = piece_image.get_rect(center=square_to_rect(square, board.turn).center)
                screen.blit(piece_image, piece_rect)

    if bot_squares is not None:
        pygame.draw.rect(screen, (0, 0, 255), square_to_rect(bot_squares[0], board.turn), 4)
        pygame.draw.rect(screen, (0, 0, 255), square_to_rect(bot_squares[1], board.turn), 4)

    # Highlight selected square
    if selected_square is not None:
        pygame.draw.rect(screen, (0, 255, 0), square_to_rect(selected_square, board.turn), 4)

    for move in legal_moves:
        pygame.draw.rect(screen, (0, 150, 0), square_to_rect(move, board.turn), 4)

    pygame.display.flip()


def draw_board_stockfish(screen, board):
    # This function will draw the chessboard with highlighted pieces and legal moves
    colors = [(210, 210, 210), (30, 30, 30)]  # Define colors for light and dark squares

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            color = colors[(col + row) % 2]
            pygame.draw.rect(screen, color, square_to_rect(square, board.turn))

            piece = board.piece_at(square)
            if piece is not None:
                scale_factor = 0.8
                piece_name = chess.piece_name(piece.piece_type).lower()
                piece_image = pygame.image.load(
                    f"images/{'black' if piece.color == chess.BLACK else 'white'}-{piece_name}.png")
                piece_image = pygame.transform.scale(piece_image,
                                                     (int(square_size * scale_factor), int(square_size * scale_factor)))

                piece_rect = piece_image.get_rect(center=square_to_rect(square, board.turn).center)
                screen.blit(piece_image, piece_rect)

    pygame.display.flip()


def square_to_rect(square, current_player):
    # Convert chess square to pixel coordinates
    file, rank = chess.square_file(square), chess.square_rank(square)

    if current_player == chess.WHITE:
        return pygame.Rect(file * square_size, (7 - rank) * square_size, square_size, square_size)
    else:
        return pygame.Rect((7 - file) * square_size, rank * square_size, square_size, square_size)


def draw_game_over(screen, winner):
    font = pygame.font.Font(None, 74)
    if winner == "You":
        text = font.render(f"Game Over! {winner} win!", True, (255, 0, 0))
    else:
        text = font.render(f"Game Over! {winner} wins!", True, (255, 0, 0))

    screen.blit(text, (150, 250))
    pygame.display.flip()
    pygame.time.wait(2000)  # Display the message for 2 seconds
    # pygame.quit()
    # sys.exit()


# ... (Your existing code)

def human_vs_bot(bot):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    bot_squares = None
    legal_moves = []
    training_examples = []
    player = 1

    while True:
        canonical_board = get_canonical_form(board, player)
        valids = get_valid_moves(canonical_board, 1)

        for event in pygame.event.get():

            if bot_squares is None:
                bef_adv = calc_advantage(board)
                bot_move = bot.make_move(board)
                board.push(bot_move)
                after_adv = -calc_advantage(board)
                adv_change = after_adv - bef_adv
                print("=======================================================================")
                print(f"This is what {bot.trainer_name}'s move score would be: " + str(adv_change))
                board.pop()
                bot_squares = (bot_move.from_square, bot_move.to_square)

            if event.type == pygame.QUIT:
                pygame.quit()
                return None
                print("Game window closed.")
            # User's Turn
            elif event.type == pygame.MOUSEBUTTONDOWN:

                col, row = event.pos[0] // square_size, event.pos[1] // square_size
                square = chess.square(col, 7 - row)

                if selected_square is None:
                    # Selecting a piece
                    if board.piece_at(square) is not None and board.piece_at(square).color == board.turn:
                        selected_square = square
                        legal_moves = [move.to_square for move in board.legal_moves if move.from_square == square]
                else:
                    # Moving the selected piece
                    if square in legal_moves:
                        trainer_move = chess.Move(selected_square, square)

                        trainer_after_adv = -calc_advantage(board)
                        trainer_adv_change = trainer_after_adv - bef_adv
                        penalty = trainer_adv_change - adv_change
                        print("This is your move score: " + str(penalty))
                        # Check for pawn promotion
                        if (
                                board.piece_at(selected_square).piece_type == chess.PAWN
                                and chess.square_rank(square) in [0, 7]
                                and chess.Move.from_uci(trainer_move.uci() + 'q') in board.legal_moves
                        ):
                            # Always promote the pawn to a queen
                            trainer_move.promotion = chess.QUEEN

                        pi = get_action_prob(canonical_board, valids, trainer_move)
                        bs, ps = zip(*get_symmetries(canonical_board, pi))
                        _, valids_sym = zip(*get_symmetries(canonical_board, valids))
                        sym = zip(bs, ps, valids_sym)

                        for b, p, valid in sym:
                            training_examples.append([to_np(b), player, p, valid])

                        action = np.random.choice(len(pi), p=pi)
                        board, player = get_next_state(board, player, action)

                        bot_squares = None

                        # Adversarial bot's move
                        # Check if the game is over due to checkmate
                        if board.is_game_over():
                            r = get_game_ended(board, player)
                            if r != 0:
                                training_examples = [(x[0], x[2], r * ((-1) ** (x[1] != player)), x[3]) for x in
                                                     training_examples]
                                winner = "You" if board.turn == chess.BLACK else "Bot"
                                draw_game_over(screen, winner)
                                pygame.quit()
                                return training_examples
                        else:
                            # Update the bot move
                            random_bot_move = np.random.choice(list(board.legal_moves))

                            pi = get_action_prob(canonical_board, valids, random_bot_move)
                            bs, ps = zip(*get_symmetries(canonical_board, pi))
                            _, valids_sym = zip(*get_symmetries(canonical_board, valids))
                            sym = zip(bs, ps, valids_sym)

                            for b, p, valid in sym:
                                training_examples.append([to_np(b), player, p, valid])

                            action = np.random.choice(len(pi), p=pi)
                            board, player = get_next_state(board, player, action)

                            selected_square = None
                            legal_moves = []

                            if board.is_game_over():
                                r = get_game_ended(board, player)
                                if r != 0:
                                    training_examples = [(x[0], x[2], r * ((-1) ** (x[1] != player)), x[3]) for x in
                                                         training_examples]
                                    winner = "You" if board.turn == chess.BLACK else "Bot"
                                    draw_game_over(screen, winner)
                                    pygame.quit()
                                    return training_examples
                    elif board.piece_at(square) is not None and board.piece_at(square).color == board.turn:
                        # Change the selected piece
                        selected_square = square
                        legal_moves = [move.to_square for move in board.legal_moves if move.from_square == square]
                    else:
                        # Deselect if an empty square or an opponent's piece is clicked
                        selected_square = None
                        legal_moves = []

        draw_board(screen, board, selected_square, legal_moves, bot_squares)
        clock.tick(144)


def stockfish_vs_bot(bot):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    board = chess.Board()
    training_examples = []
    player = 1
    stockfish = Stockfish(path=
                          r"C:\Users\sambe\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")

    while True:
        canonical_board = get_canonical_form(board, player)
        valids = get_valid_moves(canonical_board, 1)

        for event in pygame.event.get():

            bef_adv = calc_advantage(board)
            bot_move = bot.make_move(board)
            board.push(bot_move)
            after_adv = -calc_advantage(board)
            adv_change = after_adv - bef_adv
            print("=======================================================================")
            print(f"This is what the neural network's move score would be: " + str(adv_change))
            board.pop()

            if event.type == pygame.QUIT:
                pygame.quit()
                return None
                print("Game window closed.")
            # User's Turn
            else:
                stockfish.set_fen_position(board.fen())
                trainer_move = chess.Move.from_uci(stockfish.get_best_move())

                trainer_after_adv = -calc_advantage(board)
                trainer_adv_change = trainer_after_adv - bef_adv
                penalty = trainer_adv_change - adv_change
                print("This is stockfish's actual move score: " + str(penalty))
                # # Check for pawn promotion
                # if (
                #         board.piece_at(trainer_move.from_square).piece_type == chess.PAWN
                #         and chess.square_rank(trainer_move.to_square) in [0, 7]
                #         and chess.Move.from_uci(trainer_move.uci() + 'q') in board.legal_moves
                # ):
                #     # Always promote the pawn to a queen
                #     trainer_move.promotion = chess.QUEEN

                pi = get_action_prob(canonical_board, valids, trainer_move)
                bs, ps = zip(*get_symmetries(canonical_board, pi))
                _, valids_sym = zip(*get_symmetries(canonical_board, valids))
                sym = zip(bs, ps, valids_sym)

                for b, p, valid in sym:
                    training_examples.append([to_np(b), player, p, valid])

                action = np.random.choice(len(pi), p=pi)
                board, player = get_next_state(board, player, action)

                # Adversarial bot's move
                # Check if the game is over due to checkmate
                if board.is_game_over():
                    r = get_game_ended(board, player)
                    if r != 0:
                        training_examples = [(x[0], x[2], r * ((-1) ** (x[1] != player)), x[3]) for x in
                                             training_examples]
                        winner = "Stockfish" if board.turn == chess.BLACK else "Bot"
                        draw_game_over(screen, winner)

                        pygame.quit()
                        return training_examples
                else:
                    # Update the bot move
                    random_bot_move = np.random.choice(list(board.legal_moves))

                    pi = get_action_prob(canonical_board, valids, random_bot_move)
                    bs, ps = zip(*get_symmetries(canonical_board, pi))
                    _, valids_sym = zip(*get_symmetries(canonical_board, valids))
                    sym = zip(bs, ps, valids_sym)

                    for b, p, valid in sym:
                        training_examples.append([to_np(b), player, p, valid])

                    action = np.random.choice(len(pi), p=pi)
                    board, player = get_next_state(board, player, action)

                    if board.is_game_over():
                        r = get_game_ended(board, player)
                        if r != 0:
                            training_examples = [(x[0], x[2], r * ((-1) ** (x[1] != player)), x[3]) for x in
                                                 training_examples]
                            winner = "Stockfish" if board.turn == chess.BLACK else "Bot"
                            draw_game_over(screen, winner)
                            pygame.quit()
                            return training_examples

        draw_board_stockfish(screen, board)
        clock.tick(30)


def main():
    # Initialize your chess bot
    choice = input("Name of bot: ")
    bot = ChessBot(choice)

    if choice == "stockfish":
        print("You selected stockfish. Ensure that it is installed"
              " and the path in code correctly points to the executable, or it will crash.")
        num_games = input("Please provide the number of games it should play: ")
        training_examples = []
        for _ in range(int(num_games)):
            training_examples.extend(stockfish_vs_bot(bot))
    else:
        print(f"You selected {choice}. Please proceed to play alongside {choice} in the game window to help it learn.")
        training_examples = human_vs_bot(bot)

    if training_examples is None:
        print("Warning: Game prematurely exited. Aborting training procedure.")
        sys.exit()

    print("Buckle up! We're training now...")
    bot.train(training_examples, 100, 16)
    print("Training complete!")
    bot.save_model()


if __name__ == "__main__":
    main()
