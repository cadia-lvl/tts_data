import re
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# TODO: MAKE THIS MORE ROBUST, this was initially created
# using a torch text vocabulary.

graphemes = [
    'a', 'n', 'r', 'i', 's', 'l', 'u', 't', 'e', 'g',
    'k', 'm', 'ð', 'f', 'd', 'v', 'ó', 'j', 'h', 'b', 'á',
    'o', 'ö', 'p', 'æ', 'y', 'í', 'ú', 'þ', 'é', 'ý', 'x',
    'z', 'c', 'w', 'q']

grapheme_sos = '<s>'
grapheme_pad = '<pad>'
grapheme_unk = '<unk>'
g_vocab = graphemes + [grapheme_sos, grapheme_pad, grapheme_unk]

phonemes = [
    'a', 'r', 't', 's', 'n', 'ɪ', 'l', 'ʏ', 'k', 'm',
    'ð', 'ɛ', 'v', 'p', 'h', 'f', 'j', 'c', 'i', 'ɔ', 'r̥',
    'ei', 'ŋ', 'ɣ', 'ou', 'œ', 'ouː', 'au', 'ai', 'aː', 'auː',
    'iː', 'eiː', 'ɪː', 'ɛː', 'θ', 'l̥', 'tʰ', 'uː', 'aiː',
    'kʰ', 'u', 'ɔː', 'x', 'œː', 'œy', 'n̥', 'cʰ', 'œyː', 'pʰ',
    'ɲ', 'ʏː', 'ç', 'ŋ̊', 'm̥', 'ʏi', 'ɲ̊', 'ɔi']

phoneme_start = '<os>'
phoneme_end = '</os>'
phoneme_pad = '<pad>'
phoneme_unk = '<unk>'
phoneme_other = ['<unk>', '<pad>']

p_vocab = [phoneme_unk, phoneme_pad, phoneme_start, phoneme_end] + phonemes


class PronSet(Dataset):
    def __init__(
        self, graphemes, phonemes, g_vocab=g_vocab, p_vocab=p_vocab,
            reverse_input=False, **kwargs):
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
        return '''
        *********************************
        Num samples : {}
        *********************************
        '''.format(len(self.graphemes))

    def get_grapheme(self, i: int):
        # TODO : refactor this method and others to e.g. `get_word()`
        # to avoid confusion
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
        grapheme.insert(0, self.grapheme2idx[grapheme_sos])
        return torch.Tensor(grapheme).to(dtype=torch.long)

    def tensor_to_grapheme(self, tensor):
        '''
        Input arguments:
        * tensor (torch.Tensor): A
        Returns
        '''
        for i in range(tensor.shape[0]):
            print("".join(self.idx2grapheme[t.item()] for t in tensor[i, :]))

    def idxs_to_phonemes(self, idx_list):
        return " ".join(self.idx2phoneme[idx] for idx in idx_list)

    def grapheme_to_tensor(self, grapheme):
        grapheme = [self.grapheme2idx[g] for g in grapheme]
        return torch.Tensor(grapheme).to(dtype=torch.long)


    def get_phoneme(self, i: int):
        '''
        Input arguments:
        * i (int):
        '''
        phoneme = [self.phoneme2idx[p] for p in self.phonemes[i]]
        # apppend start and end tokens
        phoneme.insert(0, self.phoneme2idx[phoneme_start])
        phoneme.append(self.phoneme2idx[phoneme_end])

        return torch.Tensor(phoneme).to(dtype=torch.long)

    @property
    def num_graphemes(self):
        return len(self.g_vocab)

    @property
    def num_phonemes(self):
        return len(self.p_vocab)

    def split(self, train_r, val_r, test_r, shuffle=True):
        '''
        Split the dataset into train/validation/test sets given
        the ratios
        '''
        assert train_r + val_r + test_r == 1, "Ratios must sum to one."

        data = list(zip(self.graphemes, self.phonemes))
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
        self, bz=32, shuffle=True, num_workers=1, pin_memory=True,
            drop_last=True):
        '''def make_dataset(path:str, g_vocab):

        Return a dataloader for this dataset
        Input arguments:
        * bz (int):
        * shuffle ()
        '''
        def pad_collate(batch):
            xx = [d['grapheme'] for d in batch]
            yy = [d['phoneme'] for d in batch]
            xx_padded = pad_sequence(
                xx, batch_first=True,
                padding_value=self.grapheme2idx[grapheme_pad])
            yy_padded = pad_sequence(
                yy, batch_first=True,
                padding_value=self.phoneme2idx[phoneme_pad])

            return (
                xx_padded, yy_padded, [len(x) for x in xx],
                [len(y) for y in yy])

        return DataLoader(
            self, batch_size=bz, shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, collate_fn=pad_collate)

    def __len__(self):
        return len(self.graphemes)

    def __getitem__(self, i):
        return {
            'grapheme': self.get_grapheme(i), 'phoneme': self.get_phoneme(i)}


def normalize_grapheme(s: str) -> str:
    '''
    Input arguments:
    * s (str)
    '''
    s = s.lower().strip()  # lowercase and strip
    s = re.sub(r'\s+', ' ', s)  # collapse whitespace
    s = re.sub(r"[^{} ]".format(graphemes), '', s)
    return s


def preprocess_phoneme(p: str) -> str:
    '''
    Input arguments:
    * p (str)
    '''
    # we have to add 2 because of start ,end token padding
    p = p.lower().strip()  # lowercase and strip
    p = p.split()
    return p


def extract_pron(
        path, seperator='\t', normalize=True, g_ind=0, p_ind=1, **kwargs):
    '''
    Iterate a file where each line contains a grapheme and
    a corresponding phoneme and collects into a list of
    graphemes and phonemes.

    Input arguments:
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

    with open(path, 'r') as f:
        for line in f:
            data = line.split('\t', maxsplit=1)
            graphemes.append(
                list(normalize_grapheme(data[g_ind])) if normalize
                else list(data[g_ind]))
            phonemes.append(
                preprocess_phoneme(data[p_ind]) if normalize
                else data[p_ind])
    return graphemes, phonemes