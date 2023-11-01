import pygame
import chess


class ChessBoardDrawer:
    def __init__(self, width, height, board):
        self.width = width
        self.height = height
        self.board = board
        self.square_size = min(width, height) // 8
        self.screen = pygame.display.set_mode((width, height))

    def draw_chessboard(self):
        for rank in range(8):
            for file in range(8):
                square_color = (
                    (204, 183, 174) if (rank + file) % 2 == 0 else (112, 102, 119)
                )
                pygame.draw.rect(
                    self.screen,
                    square_color,
                    (
                        file * self.square_size,
                        rank * self.square_size,
                        self.square_size,
                        self.square_size,
                    ),
                )

    def draw_chess_pieces(self):
        piece_images = {
            "r": "images/black-rook.png",
            "n": "images/black-knight.png",
            "b": "images/black-bishop.png",
            "q": "images/black-queen.png",
            "k": "images/black-king.png",
            "p": "images/black-pawn.png",
            "R": "images/white-rook.png",
            "N": "images/white-knight.png",
            "B": "images/white-bishop.png",
            "Q": "images/white-queen.png",
            "K": "images/white-king.png",
            "P": "images/white-pawn.png",
        }

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = self.board.piece_at(square)
                if piece is not None:
                    image_path = piece_images.get(piece.symbol(), None)
                    if image_path:
                        piece_image = pygame.image.load(image_path)
                        piece_image = pygame.transform.scale(
                            piece_image, (self.square_size, self.square_size)
                        )
                        self.screen.blit(
                            piece_image,
                            (file * self.square_size, rank * self.square_size),
                        )

    def update_display(self):
        self.screen.fill((0, 0, 0))
        self.draw_chessboard()
        self.draw_chess_pieces()
        pygame.display.flip()
