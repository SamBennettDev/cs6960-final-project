import torch
import chess
import numpy as np
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
    
    def load_model(self):
        path = 'models/' + self.trainer_name + '.pth'
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def make_move(self, board):
        legal_moves = list(board.legal_moves)

        # You'll need to define your own logic to select the best move using the neural network
        best_move = self.select_best_move(board, legal_moves)

        return best_move

    def to_move(self, action):
        pi_np, v_np = action

        # Sample a move from the probability distribution
        sampled_index = np.random.choice(len(pi_np), p=pi_np)
        to_sq = sampled_index % 64
        from_sq = int(sampled_index / 64)

        return chess.Move(from_sq, to_sq)


    def select_best_move(self, board, legal_moves):
    # Convert the board and legal moves into a format suitable for your model
        board_tensor = self.board_to_tensor(board)
        valid_tensor = self.moves_to_tensor(board)

        board_tensor = board_tensor.view(1, 8, 8, 6)

        # Use the neural network to predict move values
        # predict without changing weights
        self.model.eval()
        with torch.no_grad():
            pi, v = self.model((board_tensor, valid_tensor))

        # Extract the action values from the output
        pi_np = torch.exp(pi).data.cpu().numpy()[0]
        v_np = v.data.cpu().numpy()[0]

        # Combine the action values into a tuple
        action = pi_np, v_np

        # Use the action tuple to create a move
        best_move = self.to_move(action)

        return best_move

    def from_move(self, move):
        return move.from_square*64+move.to_square

    def moves_to_tensor(self, board):        
        acts = [0] * (64*64)
        for move in board.legal_moves:
          acts[self.from_move(move)] = 1

        moves_tensor = torch.FloatTensor(np.array(acts))
        
        return moves_tensor
    
    def to_np(self, board):
        a = [0] * (8*8*6)
        for sq, pc in board.piece_map().items():
            a[sq * 6 + pc.piece_type - 1] = 1 if pc.color else -1
        return np.array(a)
        
    def board_to_tensor(self, board):
        np_board = self.to_np(board)
        board_tensor = torch.FloatTensor(np_board)

        return board_tensor





    def train(self, examples):
        """
            Train on `examples`
            
            Args:
                examples: list of examples, each example is of form 
                (board, pi, v, valids)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0

            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                target_valids = torch.FloatTensor(np.array(valids))

                # Cuda performance improvement
                boards, target_pis, target_vs, target_valids = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), target_valids.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet((boards, target_valids))
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
                log.info('({epoch}: {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Total Loss: {tl:.4f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            tl=total_loss,
                            epoch=epoch+1
                            ))
