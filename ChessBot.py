import torch
import chess
from ChessBotModel import ChessBotModel

class ChessBot:
    def __init__(self, trainer_name):
        self.trainer_name = trainer_name
        self.model = ChessBotModel()
        # check if model exists
        try:
            self.load_model()
        except:
            self.save_model()

        
    
    def save_model(self):
        path = 'models/' + self.trainer_name + '.pth'
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, model_path):
        path = 'models/' + self.trainer_name + '.pth'
        self.model.load_state_dict(path)
        self.model.eval()

    def make_move(self, board):
        legal_moves = list(board.legal_moves)

        # You'll need to define your own logic to select the best move using the neural network
        best_move = self.select_best_move(board, legal_moves)

        return best_move

    def select_best_move(self, board, legal_moves):
        # Convert the board and legal moves into a format suitable for your model
        board_tensor = self.board_to_tensor(board)
        legal_moves_tensor = self.moves_to_tensor(board)

        # Use the neural network to predict move values
        move_values = self.model(board_tensor, legal_moves_tensor)

        # Select the move with the highest predicted value
        best_move_index = torch.argmax(move_values).item()
        best_move = legal_moves[best_move_index]

        return best_move
    
    def moves_to_tensor(self, board):
        # Convert the list of legal moves to a tensor
        moves_tensor = torch.zeros(len(list(board.legal_moves)))

        for i, move in enumerate(list(board.legal_moves)):
            moves_tensor[i] = move.from_square * 64 + move.to_square

        return moves_tensor.unsqueeze(0)  # Add a batch dimension
    
    def board_to_tensor(self, board):
        # Convert the chess board state to a tensor
        # You may want to use one-hot encoding or other representations that your model expects
        # Here, we use a simple binary representation for the pieces on the board
        board_tensor = torch.zeros(12, 8, 8)  # 12 channels for 6 piece types and two colors

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_index = self.get_piece_index(piece)
                color_index = 0 if piece.color == chess.WHITE else 1
                board_tensor[piece_index + color_index * 6][square // 8][square % 8] = 1

        return board_tensor.unsqueeze(0)  # Add a batch dimension
    
    def get_piece_index(self, piece):
        # Map chess piece types to indices (0-5)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        return piece_types.index(piece.piece_type)

    def receive_human_feedback(self, feedback):
        # You can use feedback from humans to improve your bot's training or fine-tuning
        # Implement how you want to collect and utilize the feedback
        pass

    def train(self, training_data):
        # Train your neural network using the collected feedback and data
        # You'll need to define your own training pipeline based on your specific project requirements
        pass
