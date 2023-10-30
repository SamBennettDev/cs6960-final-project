import chess
from EvaluatePos import *
from ChessBot import *

board = chess.Board()
print(board.fen)


def check_game_end(board):
    if board.is_checkmate():
        if board.turn:
            return -9999 #black wins
        else:
            return 9999 #white wins

    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0

indv_scores = calc_indv_score(board)
piece_count, mat_score = calc_total_mat(board)
advantage = calc_advantage(board, indv_scores, mat_score)

print(indv_scores)
print(piece_count, mat_score)
print(advantage)

print(move_in_book(board))