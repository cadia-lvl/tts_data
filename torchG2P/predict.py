
import os
import argparse

import torch
import torch.nn as nn


from train import load_model, load_data
from G2P import G2P

def predict(model, word:str, ds, device=torch.device('cpu')):
    # TODO remove the PronSet dependency
    '''
    Input arguments:
    * model (G2P): A G2P instance, preferably pretrained
    * word (str): A single word with spaces or special tokens
    * ds (PronSet): A PronSet instance
    '''

    word_tensor = ds.grapheme_to_tensor(word).unsqueeze(dim=0)
    output = model(word_tensor.to(device), device=device).tolist()
    phoneme = ds.idxs_to_phonemes(output)
    print(phoneme)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pron_path', default='./data/prondict_ice.txt', nargs='?')
    parser.add_argument('words', default=['adolfsdóttir', 'skynsemi', 'uppvaxtarskilyrði'], nargs='?')
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