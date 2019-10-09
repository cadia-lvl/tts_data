import os
import time
import argparse

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torchtext.data as data

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


def load_model(num_g: int, num_p: int, model_path: str, **kwargs):
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
    return model


def train(
        model: G2P, train_ds: PronSet, val_ds: PronSet, ckp_path: str,
        epochs: int=10, log_step: int=100, **kwargs):
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

    model.train()
    print("Starting training.")
    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch+1, epochs))
        for idx, batch in enumerate(dl):
            output, _, _ = model(batch[0], batch[1][:, :-1].detach())
            target = batch[1][:, 1:]
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
                val_loss = validate(val_ds, model, criterion)
                print(
                    "Batch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f}"
                    .format(idx, len(dl), train_loss, val_loss))

                n_total = train_loss = 0  # reset to zero

                if val_loss < best_val_loss:
                    print("Best validation loss reached, saving model.")
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), config.best_model)


def validate(model: G2P, val_ds: PronSet, criterion, **kwargs):
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
        output, _, _ = model(batch[0], batch[1][:, :-1])
        target = batch[1][:, 1:]
        loss = criterion(output.squeeze(0), target.squeeze(0))
        val_loss += loss.item() * batch[0].shape[0]
    return val_loss / len(dl)


def main():
    parser = {
        'pron_path': './data/prondict_ice.txt',
        'exp_name': 'g2p_ice',
        'epochs': 50,
        'batch_size': 100,
        'max_len': 20,
        'beam_size': 3,
        'emb_dim': 500,
        'hidden_dim': 500,
        'log_step': 100,
        'cuda': True,
        'seed': 1337,
        'result_dir': './results',
        'data_splits': (0.9, 0.05, 0.05)}
    args = argparse.Namespace(**parser)

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

    model = load_model(
        full_ds.num_graphemes, full_ds.num_phonemes, ckp_path, **vars(args))

    print(model.summary())
    print("{} Setup Complete {}".format(*[''.join('-'*25)]*2))

    train(model, train_ds, val_ds, ckp_path, **vars(args))

if __name__ == '__main__':
    main()
