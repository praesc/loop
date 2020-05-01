# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import string
from collections import OrderedDict
import pandas as pd

import torch
from torch.autograd import Variable

from model import Loop
from data import NpzFolder
from utils import generate_merlin_wav


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
parser.add_argument('--dataset-file', type=str, default='',
                        help='CSV file')

# init
args = parser.parse_args()
if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)


def trim_pred(out, attn):
    tq = attn.abs().sum(1).data

    for stopi in range(1, tq.size(0)):
        col_sum = attn[:stopi, :].abs().sum(0).data.squeeze()
        if tq[stopi][0] < 0.5 and col_sum[-1] > 4:
            break

    out = out[:stopi, :]
    attn = attn[:stopi, :]

    return out, attn


def npy_loader_phonemes(path):
    feat = np.load(path)

    text = feat['text']

    phonemes = feat['phonemes'].astype('int64')
    phonemes = torch.from_numpy(phonemes)

    audio = feat['audio_features']
    audio = torch.from_numpy(audio)

    return phonemes, text, audio



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

    norm_path = None
    if os.path.exists(train_args.data + '/norm_info/norm.dat'):
        norm_path = train_args.data + '/norm_info/norm.dat'
    elif os.path.exists(os.path.dirname(args.checkpoint) + '/norm.dat'):
        norm_path = os.path.dirname(args.checkpoint) + '/norm.dat'
    else:
        print('ERROR: Failed to find norm file.')
        return
    train_args.noise = 0

    model = Loop(train_args)
    model.load_state_dict(weights)
    if args.gpu >= 0:
        model.cuda()
    model.eval()

    if args.spkr not in range(nspkr):
        print('ERROR: Unknown speaker id: %d.' % args.spkr)
        return

    txt, feat, spkr, output_fname = None, None, None, None
    if args.npz is not '':
        txt, text, feat = npy_loader_phonemes(args.npz)

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(torch.LongTensor([args.spkr]), volatile=True)

        fname = os.path.basename(args.npz)[:-4]
        output_fname = fname + '.gen_' + str(args.spkr)

        words = np.char.split(text).tolist()
        words = [word.encode('utf-8') for word in words]
        action = 'none'
        number = 'none'
        objectt = 'none'
        location = 'none'

        # Remove extra word for special cases
        if len(words) == 7:
            words = words[1:]

        action = words[0]
        if len(words) == 2:
            objectt = words[1]
        elif len(words) > 3:
            number = words[1]
            objectt = words[2]
            location = words[-1]


        #print(words[0], words[1], words[2], words[-1])
        #print(text)

        # Read dataframe
        frames = {}
        if os.path.exists(args.dataset_file):
            df = pd.read_csv(args.dataset_file)
            for row in zip(*[df[col].values.tolist() for col in ['path', 'speakerId', 'transcription', 'action', 'number', 'object', 'location']]):
                frames[row[0]] = {'path': row[0],
                              'speakerId': row[1],
                              'transcription': row[2],
                              'action': row[3],
                              'number': row[4],
                              'object': row[5],
                              'location': row[6]}
        
        # Add new data
        path = os.path.join('wavs/synthetic', output_fname.strip("/") + '.wav')
        frames[path] = {'path': path,
                    'speakerId': args.spkr,
                    'transcription': text,
                    'action': action,
                    'number': number,
                    'object': objectt,
                    'location': location}

        paths = []
        speakerIds = []
        transcriptions = []
        actions = []
        numbers = []
        objects = []
        locations = []
        for key, frame in frames.items():
            paths.append(frame['path'])
            speakerIds.append(frame['speakerId'])
            transcriptions.append(frame['transcription'])
            actions.append(frame['action'])
            numbers.append(frame['number'])
            objects.append(frame['object'])
            locations.append(frame['location'])

            df = pd.DataFrame(OrderedDict([('path', paths),
                            ('speakerId', speakerIds),
                            ('transcription', transcriptions),
                            ('action', actions),
                            ('number', numbers),
                            ('object', objects),
                            ('location', locations)]))
        df.to_csv(args.dataset_file)

    else:
        print('ERROR: Must supply npz file path or text as source.')
        return

    ###
    key_list = list(char2code.keys())
    val_list = list(char2code.values())
    phrase = [key_list[val_list.index(letter)] for letter in txt.data.numpy()]
    #print(phrase)
    ###


    if args.gpu >= 0:
        txt = txt.cuda()
        feat = feat.cuda()
        spkr = spkr.cuda()


    out, attn = model([txt, spkr], feat)
    out, attn = trim_pred(out, attn)

    output_dir = os.path.join(os.path.dirname(args.checkpoint), 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_merlin_wav(out.data.cpu().numpy(),
                        output_dir,
                        output_fname,
                        norm_path)

if __name__ == '__main__':
    main()
