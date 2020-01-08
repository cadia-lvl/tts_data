"""A PyTorch dataset for the G2P module"""
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

import random
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

GRAPHEMES = [
    'a', 'n', 'r', 'i', 's', 'l', 'u', 't', 'e', 'g',
    'k', 'm', 'ð', 'f', 'd', 'v', 'ó', 'j', 'h', 'b', 'á',
    'o', 'ö', 'p', 'æ', 'y', 'í', 'ú', 'þ', 'é', 'ý', 'x',
    'z', 'c', 'w', 'q']

GRAPHEME_SOS = '<s>'
GRAPHEME_PAD = '<pad>'
GRAPHEME_UNK = '<unk>'
G_VOCAB = GRAPHEMES + [GRAPHEME_SOS, GRAPHEME_PAD, GRAPHEME_UNK]

PHONEMES = [
    'a', 'r', 't', 's', 'n', 'ɪ', 'l', 'ʏ', 'k', 'm',
    'ð', 'ɛ', 'v', 'p', 'h', 'f', 'j', 'c', 'i', 'ɔ', 'r̥',
    'ei', 'ŋ', 'ɣ', 'ou', 'œ', 'ouː', 'au', 'ai', 'aː', 'auː',
    'iː', 'eiː', 'ɪː', 'ɛː', 'θ', 'l̥', 'tʰ', 'uː', 'aiː',
    'kʰ', 'u', 'ɔː', 'x', 'œː', 'œy', 'n̥', 'cʰ', 'œyː', 'pʰ',
    'ɲ', 'ʏː', 'ç', 'ŋ̊', 'm̥', 'ʏi', 'ɲ̊', 'ɔi']

PHONEME_START = '<os>'
PHONEME_END = '</os>'
PHONEME_PAD = '<pad>'
PHONEME_UNK = '<unk>'
PHONEME_OTHER = ['<unk>', '<pad>']

P_VOCAB = [PHONEME_UNK, PHONEME_PAD, PHONEME_START, PHONEME_END] + PHONEMES


class PronSet(Dataset):
    """PyTorch Dataset for the G2P module"""
    def __init__(
            self, graphemes, phonemes, g_vocab=G_VOCAB, p_vocab=P_VOCAB,
            reverse_input=False):
        '''
        Input arguments:
        * graphemes (list): graphemes[i] is the list of graphemes of the i-th
        sample
        * phonemes (list): phonemes[i] is the list of phonemes of the i-th
        sample
        * g_vocab (list): The grapheme vocabulary, including special tokens
        * p_vocab (list): The phoneme vocabulary, including special tokens
        * reverse_input (bool=False): If set to True, the grapheme will be
        reversed in the input.
        '''
        assert len(graphemes) == len(phonemes),\
            "Number of graphemes and phonemes must match"

        self.graphemes = graphemes
        self.phonemes = phonemes

        self.phoneme2idx = {p_vocab[i]: i for i in range(len(p_vocab))}
        self.idx2phoneme = {v: k for k, v in self.phoneme2idx.items()}

        self.grapheme2idx = {g_vocab[i]: i for i in range(len(g_vocab))}
        self.idx2grapheme = {v: k for k, v in self.grapheme2idx.items()}

        self.g_vocab = g_vocab
        self.p_vocab = p_vocab
        self.reverse_input = reverse_input

    def summary(self):
        """Returns a summary of this dataset in text form"""
        return '''
        *********************************
        Num samples : {}
        *********************************
        '''.format(len(self.graphemes))

    def get_grapheme(self, i: int):
        '''
        Input arguments:
        * i (int): The index of the grapheme to fetch

        Returns: g
        * g (torch.Tensor): A (g_seq x 1) tensor containing the
        grapheme indexes of the fetched grapheme
        '''
        grapheme = [self.grapheme2idx[g] for g in self.graphemes[i]]
        if self.reverse_input:
            grapheme = grapheme[::-1]
        # apppend start-of-sentence-tokene
        grapheme.insert(0, self.grapheme2idx[GRAPHEME_SOS])
        return torch.Tensor(grapheme).to(dtype=torch.long)

    def tensor_to_grapheme(self, tensor):
        """Converts a one-hot encoded tensor into a grapheme"""
        for i in range(tensor.shape[0]):
            print("".join(self.idx2grapheme[t.item()] for t in tensor[i, :]))

    def idxs_to_phonemes(self, idx_list):
        """Converts a list of phoneme vocabulary index values into the
        corresponding grapheme"""
        return " ".join(self.idx2phoneme[idx] for idx in idx_list)

    def grapheme_to_tensor(self, grapheme):
        """Converts a grapheme into a one-hot encoded tensor"""
        grapheme = [self.grapheme2idx[g] for g in grapheme]
        return torch.Tensor(grapheme).to(dtype=torch.long)

    def get_phoneme(self, i: int):
        """Gets the i-th phoneme from the dataset"""
        phoneme = [self.phoneme2idx[p] for p in self.phonemes[i]]
        # apppend start and end tokens
        phoneme.insert(0, self.phoneme2idx[PHONEME_START])
        phoneme.append(self.phoneme2idx[PHONEME_END])

        return torch.Tensor(phoneme).to(dtype=torch.long)

    @property
    def num_graphemes(self):
        """Returns the number of graphemes in the grapheme vocabulary"""
        return len(self.g_vocab)

    @property
    def num_phonemes(self):
        """Returns the number of phones in the phoneme vocabulary"""
        return len(self.p_vocab)

    def split(self, train_r: float, val_r: float, test_r: float, shuffle=True):
        '''
        Split the dataset into train/validation/test sets given
        the ratios

        Input arguments:
        * train_r (float): The ratio of the data that should be training data
        * val_r (float): The ratio of the data that should be validation data
        * test_r (float): The ratio of the data that should be test data
        * shuffle (bool): If True, the data is shuffled before splitting

        '''
        assert train_r + val_r + test_r == 1, "Ratios must sum to one."

        data = list(zip(self.graphemes, self.phonemes))
        if shuffle:
            random.shuffle(data)

        train_g, train_p = zip(*data[0:int(len(data)*train_r)])
        val_g, val_p = zip(
            *data[int(len(data)*train_r):int(len(data)*(train_r+val_r))])
        test_g, test_p = zip(*data[int(len(data)*(train_r+val_r)):])

        return (
            PronSet(
                train_g, train_p, g_vocab=self.g_vocab, p_vocab=self.p_vocab,
                reverse_input=self.reverse_input),
            PronSet(
                val_g, val_p, g_vocab=self.g_vocab, p_vocab=self.p_vocab,
                reverse_input=self.reverse_input),
            PronSet(
                test_g, test_p, g_vocab=self.g_vocab, p_vocab=self.p_vocab,
                reverse_input=self.reverse_input))

    def get_loader(
            self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 1,
            pin_memory: bool = True, drop_last=True):
        '''
        Return a dataloader for this dataset
        Input arguments:
        * batch_size (int): The batch size
        * shuffle (bool): If True, the data is shuffled before batching
        * num_workers (int): The number of dataloader workers
        * pin_memory (bool): If True, memory for dataloader is pinned
        * drop_last (bool): If True, the last batch is dropped if smaller
        than batch_size
        '''
        def pad_collate(batch):
            batch_in = [d['grapheme'] for d in batch]
            batch_out = [d['phoneme'] for d in batch]
            xx_padded = pad_sequence(
                batch_in, batch_first=True,
                padding_value=self.grapheme2idx[GRAPHEME_PAD])
            yy_padded = pad_sequence(
                batch_out, batch_first=True,
                padding_value=self.phoneme2idx[PHONEME_PAD])

            return (
                xx_padded, yy_padded, [len(x) for x in batch_in],
                [len(y) for y in batch_out])

        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, collate_fn=pad_collate)

    def __len__(self):
        return len(self.graphemes)

    def __getitem__(self, i):
        return {
            'grapheme': self.get_grapheme(i), 'phoneme': self.get_phoneme(i)}


def normalize_grapheme(grapheme: str) -> str:
    '''
    Normalize a grapheme

    Input arguments:
    * grapheme (str): The grapheme to normalize
    '''
    grapheme = grapheme.lower().strip()  # lowercase and strip
    grapheme = re.sub(r'\s+', ' ', grapheme)  # collapse whitespace
    grapheme = re.sub(r"[^{} ]".format(GRAPHEMES), '', grapheme)
    return grapheme


def preprocess_phoneme(phoneme: str) -> str:
    '''
    Normalize a phoneme

    Input arguments:
    * phoneme (str): The phoneme to normalize
    '''
    # we have to add 2 because of start ,end token padding
    phoneme = phoneme.lower().strip()  # lowercase and strip
    phoneme = phoneme.split()
    return phoneme


def extract_pron(
        path: str, seperator: str = '\t', normalize: str = True,
        g_ind: int = 0, p_ind: int = 1):
    '''
    Iterate a file where each line contains a grapheme and
    a corresponding phoneme and collects into a list of
    graphemes and phonemes.

    Input arguments:
    * path (str): File path
    * seperator (str='\t')
    * normalize (bool=True)
    * g_ind (int=0)
    * p_ind (int=1)

    Returns two list,
    * graphemes[i] is a list of the corresponding graphemes in the i-th word
    * phonemes[i] is a list longof the corresponding phonemes in the i-th word
    '''
    graphemes = []
    phonemes = []

    with open(path, 'r') as in_file:
        for line in in_file:
            data = line.split(seperator, maxsplit=1)
            graphemes.append(
                list(normalize_grapheme(data[g_ind])) if normalize
                else list(data[g_ind]))
            phonemes.append(
                preprocess_phoneme(data[p_ind]) if normalize
                else data[p_ind])
    return graphemes, phonemes
