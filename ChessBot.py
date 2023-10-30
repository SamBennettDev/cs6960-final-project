import chess.polyglot

def move_in_book(board):
    try:
        move = chess.polyglot.MemoryMappedReader("./human.bin").weighted_choice(board).move
        return move
    except:
        return 'none'