import chess
from PieceTables import *


# evaluates total material
def calc_total_mat(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    mat_score = (
        (100 * (wp - bp))
        + (300 * (wn - bn))
        + (310 * (wb - bb))
        + (500 * (wr - br))
        + (900 * (wq - bq))
    )

    return {
        "wp": wp,
        "bp": bp,
        "wn": wn,
        "bn": bn,
        "wb": wb,
        "bb": bb,
        "wr": wr,
        "br": br,
        "wq": wq,
        "bq": bq,
    }, mat_score


# evaluates the positioning of pieces on the board
def calc_indv_score(board):
    wp_score = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    bp_score = sum(
        [
            -pawntable[chess.square_mirror(i)]
            for i in board.pieces(chess.PAWN, chess.BLACK)
        ]
    )

    wk_score = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    bk_score = sum(
        [
            -knightstable[chess.square_mirror(i)]
            for i in board.pieces(chess.KNIGHT, chess.BLACK)
        ]
    )

    wb_score = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bb_score = sum(
        [
            -bishopstable[chess.square_mirror(i)]
            for i in board.pieces(chess.BISHOP, chess.BLACK)
        ]
    )

    wr_score = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    br_score = sum(
        [
            -rookstable[chess.square_mirror(i)]
            for i in board.pieces(chess.ROOK, chess.BLACK)
        ]
    )

    wq_score = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    bq_score = sum(
        [
            -queenstable[chess.square_mirror(i)]
            for i in board.pieces(chess.QUEEN, chess.BLACK)
        ]
    )

    wki_score = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    bki_score = sum(
        [
            -kingstable[chess.square_mirror(i)]
            for i in board.pieces(chess.KING, chess.BLACK)
        ]
    )

    return {
        "wp": wp_score,
        "bp": bp_score,
        "wk": wk_score,
        "bk": bk_score,
        "wb": wb_score,
        "bb": bb_score,
        "wr": wr_score,
        "br": br_score,
        "wq": wq_score,
        "bq": bq_score,
        "wki": wki_score,
        "bki": bki_score,
    }


def calc_advantage(board):
    indv_scores = calc_indv_score(board)
    mat_score = calc_total_mat(board)[1]
    p_score = indv_scores["wp"] + indv_scores["bp"]
    k_score = indv_scores["wk"] + indv_scores["bk"]
    b_score = indv_scores["wb"] + indv_scores["bb"]
    r_score = indv_scores["wr"] + indv_scores["br"]
    q_score = indv_scores["wq"] + indv_scores["bq"]
    ki_score = indv_scores["wki"] + indv_scores["bki"]

    advantage = mat_score + p_score + k_score + b_score + r_score + q_score + ki_score

    if board.turn:
        return advantage
    else:
        return -advantage
