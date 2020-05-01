# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
from phonemizer import phonemize
from phonemizer import separator
import string

import torch

from model import Loop


parser = argparse.ArgumentParser(description='PyTorch Phonological Loop \
                                    Generation')
parser.add_argument('--npz', type=str, default='',
                    help='Dataset sample to generate.')
parser.add_argument('--text', default='',
                    type=str, help='Free text to generate.')
parser.add_argument('--spkr', default=0,
                    type=int, help='Speaker id.')
parser.add_argument('--checkpoint', default='checkpoints/vctk/lastmodel.pth',
                    type=str, help='Model used for generation.')
parser.add_argument('--gpu', default=-1,
                    type=int, help='GPU device ID, use -1 for CPU.')
parser.add_argument('--out-dir', default='',
                    type=str, help='Output directory.')

# init
args = parser.parse_args()

def text2phone(text, char2code):
    seperator = separator.Separator('', '', ' ')
    ph = phonemize.phonemize(text, separator=seperator)
    ph = ph.split(' ')
    ph.remove('')

    result = [char2code[p] for p in ph]
    return torch.LongTensor(result)


def main():
    weights = torch.load(args.checkpoint,
                         map_location=lambda storage, loc: storage)
    opt = torch.load(os.path.dirname(args.checkpoint) + '/args.pth')
    train_args = opt[0]

    char2code = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5,  'ay': 6,
                 'b': 7, 'ch': 8, 'd': 9, 'dh': 10, 'eh': 11, 'er': 12, 'ey': 13,
                 'f': 14, 'g': 15, 'hh': 16, 'i': 17, 'ih': 18, 'iy': 19, 'jh': 20,
                 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'ng': 25, 'ow': 26, 'oy': 27,
                 'p': 28, 'pau': 29, 'r': 30, 's': 31, 'sh': 32, 'ssil': 33,
                 't': 34, 'th': 35, 'uh': 36, 'uw': 37, 'v': 38, 'w': 39, 'y': 40,
                 'z': 41}
    nspkr = train_args.nspk   

    if args.spkr not in range(nspkr):
        print('ERROR: Unknown speaker id: %d.' % args.spkr)
        return

    txt, feat, spkr, output_fname = None, None, None, None
    if args.text is not '':
        txt = text2phone(args.text, char2code)
        feat = torch.FloatTensor(txt.size(0)*20, 63)
        spkr = torch.LongTensor([args.spkr])

        #Store
        output_dir = os.path.join(args.out_dir, 'speaker' + str(args.spkr))
        os.makedirs(output_dir, exist_ok=True)
        output_name = str(args.text).replace(" ", "_")
        np.savez(os.path.join(output_dir, output_name), phonemes=txt.numpy(), audio_features=feat.numpy(), text=args.text)

    else:
        print('ERROR: Must supply text as source.')
        return

if __name__ == '__main__':
    main()
