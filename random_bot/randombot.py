from random import randint

class RandomBot:
    
    def get_move(self, pos, tleft):
        lmoves = pos.legal_moves()
        rm = randint(0, len(lmoves)-1)
        return lmoves[rm]

