from __future__ import print_function
from args import get_args
from trainer import Trainer

if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    trainer.train()

#python train.py -T seg --style swd -D G C --ex G2C_swd
#python train.py -T clf --style gram -D M MM --ex M2MM_gram
