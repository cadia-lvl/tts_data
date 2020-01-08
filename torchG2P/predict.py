"""A prediction function for the PyTorch G2P model"""
# -*- coding: utf-8 -*-

# Copyright 2020 Atli Thor Sigurgeirsson <atlithors@ru.is>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch

from train import load_data, load_model


def predict(model, word: str, dataset, device=torch.device('cpu')):
    '''
    Input arguments:
    * model (G2P): A G2P instance, preferably pretrained
    * word (str): A single word with spaces or special tokens
    * dataset (PronSet): A PronSet instance
    '''

    word_tensor = dataset.grapheme_to_tensor(word).unsqueeze(dim=0)
    output = model(word_tensor.to(device), device=device).tolist()
    phoneme = dataset.idxs_to_phonemes(output)
    print(phoneme)


def main():
    """Argument parser for making G2P predictions"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pron_path', default='./data/prondict_ice.txt',
        nargs='?')
    parser.add_argument(
        'words', default=['adolfsdóttir', 'skynsemi', 'uppvaxtarskilyrði'],
        nargs='?')
    parser.add_argument('exp_name', default='g2p_ice', nargs='?')
    parser.add_argument('emb_dim', default=500, nargs='?')
    parser.add_argument('hidden_dim', default=500, nargs='?')
    parser.add_argument('cuda', default=True, nargs='?')
    parser.add_argument('seed', default=1337, nargs='?')
    parser.add_argument('result_dir', default='./results', nargs='?')
    parser.add_argument('data_splits', default=(0.9, 0.05, 0.05), nargs='?')
    args = parser.parse_args()

    exp_dir = os.path.join(args.result_dir, args.exp_name)
    ckp_path = os.path.join(exp_dir, 'mdl.ckpt')

    full_ds, _ = load_data(
        args.pron_path, args.data_splits, **vars(args))
    model = load_model(
        full_ds.num_graphemes, full_ds.num_phonemes, ckp_path, **vars(args))

    for word in args.words:
        print(word)
        predict(model, word, full_ds)


if __name__ == '__main__':
    main()
