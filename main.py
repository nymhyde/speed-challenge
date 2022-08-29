#!/usr/bin/env python3

import cv2
import os
import sys
import argparse

from solver import Solver


def main(params):
    solver = Solver(params)
    hf_factor = solver.run()
#    solver.test(hf_factor, params.save_txt)
    cv2.destroyAllWindows()





if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_vid', type=str, default='data/train.mp4', help='train video file path')
    parser.add_argument('--test_vid', type=str, default='data/test.mp4', help='test video file path')
    parser.add_argument('--train_txt', type=str, default='data/train.txt', help='train txt file path')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--len_gt', type=int, default=20400, help='length of ground truth labels')
    parser.add_argument('--save_txt', type=bool, default=True, help='Toggle to save test prediction to a file')

    params = parser.parse_args()
    print(params)
    main(params)
