import numpy as np
from random import randint
from copy import deepcopy

class RolloutBot:
    
    def get_move(self, pos, tleft):
        # with open('temp.txt', 'w') as f:
        #     f.write('BOARD:\n')
        #     f.write(str(type(pos.board)))
        #     f.write(str(pos.board))
        #     f.write('\nMACROBOARD:\n')
        #     f.write(str(type(pos.macroboard)))
        #     f.write(str(pos.macroboard))
        #     f.write('\n')
        #     for move in pos.legal_moves():
        #         f.write(str(move))
        #         f.write('\n')
        #         new_pos = deepcopy(pos)
        #         new_pos.make_move(move[0], move[1], self.myid)
        #         f.write(str(new_pos.board))
        #         f.write('\n')
        #         f.write(str(self.score_output(new_pos)))
        #         f.write('\n')
        best_move = None
        best_move_score = np.NINF
        for move in pos.legal_moves():
            new_pos = deepcopy(pos)
            new_pos.make_move(move[0], move[1], self.myid)
            score = self.score_output(new_pos)
            if score >= best_move_score:
                best_move_score = score
                best_move = move
        if best_move:
            return best_move
        lmoves = pos.legal_moves()
        rm = randint(0, len(lmoves)-1)
        return lmoves[rm]

    def init_weights(self):
        conv_layer_output_size = 5
        self.conv_weights = np.random.rand(9*3, conv_layer_output_size)
        self.score_weights = np.random.rand(conv_layer_output_size*9+9, 1)

    def calc_board_vec(self, pos):
        npboard = np.array(pos.board)
        myboard = (npboard == self.myid).astype(int)
        oppboard = (npboard == self.oppid).astype(int)
        emptyboard = (npboard == 0).astype(int)
        board_vec = np.empty(9*9*3)
        board_vec[::3] = myboard
        board_vec[1::3] = oppboard
        board_vec[2::3] = emptyboard
        return board_vec

    def conv_output(self, pos):
        board_vec = self.calc_board_vec(pos)
        return np.maximum(0, board_vec.reshape(-1, 27).dot(self.conv_weights).reshape(-1))

    def score_output(self, pos):
        conved = self.conv_output(pos)
        npmacroboard = (np.array(pos.macroboard) == -1).astype(int)
        return np.concatenate((conved, npmacroboard)).dot(self.score_weights)

