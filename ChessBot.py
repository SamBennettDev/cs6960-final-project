import logging
import time

import torch
import chess
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ChessBotModel import ChessBotModel
from utils import AverageMeter

writer = SummaryWriter()
# set up logging to file
logging.basicConfig(level=logging.DEBUG)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
log = logging.getLogger(__name__)


def to_move(action):
    to_sq = action % 64
    from_sq = int(action / 64)
    return chess.Move(from_sq, to_sq)


def from_move(move):
    return move.from_square * 64 + move.to_square

def to_np(board):
    a = [0] * (8 * 8 * 6)
    for sq, pc in board.piece_map().items():
        a[sq * 6 + pc.piece_type - 1] = 1 if pc.color else -1
    return np.array(a)

class ChessBot(object):
    def __init__(self, trainer_name):
        super(ChessBot, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        print("=======================================================================")
        print(f"Using CUDA: {torch.cuda.is_available()}")
        print("=======================================================================")


        self.trainer_name = trainer_name
        self.model = ChessBotModel()
        # check if model exists
        try:
            self.load_model()
        except:
            self.save_model()
        if self.use_cuda:
            self.model.cuda()

    def save_model(self):
        path = 'models/' + self.trainer_name + '.pth'
        print(f"Saving model {self.trainer_name} as {path}.")
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        path = 'models/' + self.trainer_name + '.pth'
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def make_move(self, board):
        legal_moves = list(board.legal_moves)

        # You'll need to define your own logic to select the best move using the neural network
        best_move = self.select_best_move(board, legal_moves)

        return best_move

    def select_best_move(self, board, legal_moves):
        # Convert the board and legal moves into a format suitable for your model
        board_tensor = self.board_to_tensor(board)
        valid_tensor = self.moves_to_tensor(board)

        if self.use_cuda:
            board_tensor = board_tensor.contiguous().cuda()
            valid_tensor = valid_tensor.contiguous().cuda()

        board_tensor = board_tensor.view(1, 8, 8, 6)

        # Use the neural network to predict move values
        # predict without changing weights
        self.model.eval()
        with torch.no_grad():
            pi, v = self.model((board_tensor, valid_tensor))

        pi_np = torch.exp(pi).data.cpu().numpy()[0]
        v_np = v.data.cpu().numpy()[0]
        translated = []

        for move in legal_moves:
            translated.append(from_move(move))

        highest_score = translated[0]

        for move in translated[1:]:
            if pi_np[move] > pi_np[highest_score]:
                highest_score = move

        return to_move(highest_score)

    def moves_to_tensor(self, board):
        acts = [0] * (64 * 64)
        for move in board.legal_moves:
            acts[from_move(move)] = 1

        moves_tensor = torch.FloatTensor(np.array(acts))

        return moves_tensor

    def board_to_tensor(self, board):
        np_board = to_np(board)
        board_tensor = torch.FloatTensor(np_board)

        return board_tensor

    def loss_pi(self, targets, outputs):
        """Custom loss function for probabilty distribuition"""
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """Custom loss function for scalar value"""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def train(self, examples, epochs, batch_size):
        """
            Train on `examples`
            
            Args:
                examples: list of examples, each example is of form 
                (board, pi, v, valids)
        """
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.model.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0

            while batch_idx < int(len(examples) / batch_size):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                target_valids = torch.FloatTensor(np.array(valids))

                if self.use_cuda:
                    # Cuda performance improvement
                    boards, target_pis, target_vs, target_valids = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), target_valids.contiguous().cuda()
                else:
                    boards, target_pis, target_vs, target_valids = boards.contiguous(), target_pis.contiguous(), target_vs.contiguous(), target_valids.contiguous()


                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.model((boards, target_valids))
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                writer.add_scalar("Loss/train", l_pi.item(), batch_idx)
                writer.add_scalar("Loss/train", l_v.item(), batch_idx)
                writer.flush()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                log.info(
                    '({epoch}: {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Total Loss: {tl:.4f}'.format(
                        batch=batch_idx,
                        size=int(len(examples) / batch_size),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        lpi=pi_losses.avg,
                        lv=v_losses.avg,
                        tl=total_loss,
                        epoch=epoch + 1
                    ))
