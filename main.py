import os, sys, tqdm, argparse, time
import numpy as np

import lib
from tasks import classification
from tasks import segmentation as seg
from tasks import detection as det

import config as cfg

def main():
    args = cfg.global_parse_args()
    classification.train.main(args)
    
    classification.test.main(args)



if __name__=="__main__":
    main()