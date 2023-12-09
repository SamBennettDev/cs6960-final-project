import chess
import chess.svg
from ChessBot import ChessBot
import pygame
import sys

square_size = 75


def draw_board(screen, board, selected_square, legal_moves):
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


    # Highlight selected square
    if selected_square is not None:
        pygame.draw.rect(screen, (0, 255, 0), square_to_rect(selected_square, board.turn), 4)

    for move in legal_moves:
        pygame.draw.rect(screen, (0, 150, 0), square_to_rect(move, board.turn), 4)

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
    pygame.quit()
    sys.exit()


# ... (Your existing code)

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    legal_moves = []

    # Initialize your chess bot
    enemy_bot = ChessBot("stockfish")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
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
                        move = chess.Move(selected_square, square)
                        # Check if the move is legal and does not involve moving the king
                        # Check if the move is legal and does not involve moving the king
                        # Check if the move is legal and does not involve moving the king
                        # if move in board.legal_moves and board.piece_at(selected_square).piece_type != chess.KING:
                            # Check for pawn promotion
                        if (
                                board.piece_at(selected_square).piece_type == chess.PAWN
                                and chess.square_rank(square) in [0, 7]
                                and chess.Move.from_uci(move.uci() + 'q') in board.legal_moves
                        ):
                            # Always promote the pawn to a queen
                            move.promotion = chess.QUEEN

                        print(f"Attempting to push move: {move}")
                        board.push(move)

                        # Check if the game is over due to checkmate
                        if board.is_checkmate():
                            winner = "You" if board.turn == chess.BLACK else "Bot"
                            draw_game_over(screen, winner)
                        else:
                            # Update the bot move
                            bot_move = enemy_bot.make_move(board)
                            board.push(bot_move)

                            selected_square = None
                            legal_moves = []

                            # Check if the game is over
                            if board.is_game_over():
                                winner = "You" if board.turn == chess.BLACK else "Bot"
                                draw_game_over(screen, winner)
                    # else:
                        #     print(f"Illegal move: {move}")
                    elif board.piece_at(square) is not None and board.piece_at(square).color == board.turn:
                        # Change the selected piece
                        selected_square = square
                        legal_moves = [move.to_square for move in board.legal_moves if move.from_square == square]
                    else:
                        # Deselect if an empty square or an opponent's piece is clicked
                        selected_square = None
                        legal_moves = []

        draw_board(screen, board, selected_square, legal_moves)
        clock.tick(30)


if __name__ == "__main__":
    main()
