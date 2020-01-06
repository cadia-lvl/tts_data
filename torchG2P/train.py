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

import os
import time
import argparse

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import Levenshtein


from PronSet import PronSet, extract_pron
from G2P import G2P


def load_data(path: str, splits: List[float], **kwargs):
    '''
    Input arguments:
    * splits (List[float]): The ratios for train/validation/test
    sets to use from the complete dataset

    Returns: train_ds, val_ds, test_ds
    * train_ds: (PronSet): The training dataset
    * val_ds: (PronSet): The validation dataset
    * test_ds: (PronSet): The test dataset
    * **kwargs: Any number of additional keyword arguments passed to
    PronSet.PronSet or PronSet.extract_pron
    '''
    graphemes, phonemes = extract_pron('./data/prondict_ice.txt', **kwargs)
    ds = PronSet(graphemes, phonemes, **kwargs)
    return ds, ds.split(*splits)


def load_model(
    num_g: int, num_p: int, model_path: str, device=torch.device('cpu'),
    **kwargs):
    '''
    Input arguments:
    * num_g (int): The size of the grapheme vocabulary
    * num_p (int): The size of the phoneme voccabulary
    * model_path (str) : Where the model checkpoint is saved
    * **kwargs : Any number of keyword arguments. This should ideally
    contain the G2P keyword arguments.

    Returns:
    * model (G2P): A G2P instance with the given model configuration. If a
    checkpoint is found at <path>, the model loads the state dictionary.
    '''
    model = G2P(num_g, num_p, **kwargs)
    if os.path.isfile(model_path):
        print("A pretrained model will be loaded from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    else:
        print("A new model will be created.")
    return model.to(device)


def train(
        model: G2P, train_ds: PronSet, val_ds: PronSet, ckp_path: str,
        epochs: int=10, log_step: int=100, device=torch.device('cpu'), **kwargs):
    '''
    Input arguments:
    * model (G2P): A G2P instance
    * train_ds (PronSet): The training dataset
    * val_ds (PronSet): The validation dataset
    * ckp_path (str): The path for storing the model
    * epoch (int): The number of epochs to train
    * log_step (int): The interval of logging during training
    '''
    dl = train_ds.get_loader()
    criterion = nn.NLLLoss()  # TODO: add to parameters
    optimizer = optim.Adam(model.parameters())

    n_total = 0  # number of samples
    train_loss = 0.0
    best_val_loss = float('inf')

    print("Starting training.")
    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch+1, epochs))
        for idx, batch in enumerate(dl):
            output, _, _ = model(
                batch[0].to(device), batch[1][:, :-1].detach().to(device), device=device)
            target = batch[1][:, 1:].to(device)
            loss = criterion(
                output.view(output.shape[0] * output.shape[1], -1),
                target.contiguous().view(target.shape[0] * target.shape[1]))
            optimizer.zero_grad()
            loss.backward()
            # TODO: add to params
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.3, 'inf')
            optimizer.step()

            n_total += batch[0].shape[0]  # num samples in batch
            # the NLLLoss returns a scalar average over all samples in the
            # batch if reduction='none'
            train_loss += loss.item() * batch[0].shape[0]

            if (idx + 1) % log_step == 0:
                train_loss /= n_total  # compute the average so far
                val_loss = validate(model, val_ds, criterion, device=device)
                print(
                    "Batch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f}"
                    .format(idx, len(dl), train_loss, val_loss))

                n_total = train_loss = 0  # reset to zero

                if val_loss < best_val_loss:
                    print("Best validation loss reached, saving model.")
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), ckp_path)


def validate(
        model: G2P, val_ds: PronSet, criterion, device=torch.device('cpu'),
        **kwargs):
    '''
    Input arguments:
    * model (G2P): A G2P instance
    * val_ds (PronSet): The validation dataset
    * criterion (Torch criterion): The same loss criteria as used
    for training

    Returns:
    The validation loss
    '''
    val_loss = 0
    dl = val_ds.get_loader(bz=1)
    val_len = len(dl)

    model.eval()
    print("Starting Evaluation")
    for idx, batch in enumerate(dl):
        output, _, _ = model(batch[0].to(device), batch[1][:, :-1].to(device), device=device)
        target = batch[1][:, 1:].to(device)
        loss = criterion(output.squeeze(0), target.squeeze(0))
        val_loss += loss.item() * batch[0].shape[0]
    model.train()
    return val_loss / len(dl)


def test(model, test_ds, device=torch.device('cpu'), **kwargs):
    def phoneme_error_rate(p_seq1, p_seq2):
        p_vocab = set(p_seq1 + p_seq2)
        p2c = dict(zip(p_vocab, range(len(p_vocab))))
        c_seq1 = [chr(p2c[p]) for p in p_seq1]
        c_seq2 = [chr(p2c[p]) for p in p_seq2]
        return Levenshtein.distance(''.join(c_seq1),
                                    ''.join(c_seq2)) / len(c_seq2)

    test_per = test_wer = 0
    dl = test_ds.get_loader(bz=1)
    for idx, batch in enumerate(dl):
        output = model(batch[0].to(device), device=device).tolist()
        target = batch[1][:, 1:].to(device).squeeze(0).tolist()
        # calculate per, wer here
        per = phoneme_error_rate(output, target)
        wer = int(output != target)
        test_per += per  # batch_size = 1
        test_wer += wer

    test_per = test_per / len(dl) * 100
    test_wer = test_wer / len(dl) * 100
    print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
          .format(test_per, test_wer))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pron_path', default='./data/prondict_ice.txt', nargs='?')
    parser.add_argument('exp_name', default='g2p_ice', nargs='?')
    parser.add_argument('epochs', default=50, nargs='?')
    parser.add_argument('batch_size', default=100, nargs='?')
    parser.add_argument('max_len', default=20, nargs='?')
    parser.add_argument('beam_size', default=3, nargs='?')
    parser.add_argument('emb_dim', default=500, nargs='?')
    parser.add_argument('hidden_dim', default=500, nargs='?')
    parser.add_argument('log_step', default=2000, nargs='?')
    parser.add_argument('cuda', default=True, nargs='?')
    parser.add_argument('seed', default=1337, nargs='?')
    parser.add_argument('result_dir', default='./results', nargs='?')
    parser.add_argument('data_splits', default=(0.9, 0.05, 0.05), nargs='?')
    args = parser.parse_args()

    print("{} Starting Setup {}".format(*[''.join('-'*25)]*2))
    # CUDA and reproducability stuff
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device('cpu')
    torch.manual_seed(args.seed)
    if args.cuda:
        print("A cuda device is available and will be used")
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        print("CPU will be used for inference")
    # Make sure directories exist
    exp_dir = os.path.join(args.result_dir, args.exp_name)
    ckp_path = os.path.join(exp_dir, 'mdl.ckpt')
    log_path = os.path.join(exp_dir, 'run.log')  # TODO: not used currently
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    full_ds, (train_ds, val_ds, test_ds) = load_data(
        args.pron_path, args.data_splits, **vars(args))
    print("Batch size: {}".format(args.batch_size))
    print(train_ds.summary())
    print(val_ds.summary())
    print(test_ds.summary())

    model = load_model(
        full_ds.num_graphemes, full_ds.num_phonemes, ckp_path, **vars(args))

    print(model.summary())
    print("{} Setup Complete {}".format(*[''.join('-'*25)]*2))

    train(model, train_ds, val_ds, ckp_path, **vars(args))

    test(model, test_ds, **vars(args))


if __name__ == '__main__':
    main()