import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torchtext.data as data

from postprocess import Beam


class G2P(nn.Module):
    def __init__(
            self, num_g: int, num_p: int, emb_dim: int=256,
            hidden_dim: int=256, beam_size: int=2, max_decode_len: int=20,
            device=torch.device('cpu'), **kwargs):
        '''
        Input arguments:
        * num_g (int): The size of the grapheme vocabulary
        * num_p (int): The size of the phoneme voccabulary
        * emb_dim (int): The dimensionality of the character embeddings used
        * hidden_dim (int): The hidden dimensionality of the RNNs
        * device (torch.device=cpu): The device used to store the model
        * beam_size (int=2): The size of decoding beams
        * max_decode_len (int=20): The maximum decoding output length
        '''
        super(G2P, self).__init__()
        self.encoder = Encoder(num_g,  emb_dim, hidden_dim, device=device)
        self.decoder = Decoder(num_p, emb_dim, hidden_dim)

        self.num_g = num_g
        self.num_p = num_p
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.device = device
        self.beam_size = beam_size
        self.max_decode_len = max_decode_len

    def summary(self) -> str:
        return '''
        *********************************
        G2P model summary

        Grapheme Embedding : in/out {}/{}
        Encoder RNN : in/hidden {}/{}
        Phoneme Embedding : in/out {}/{}
        Decoder RNN : in/hidden {}/{}

        device : {}
        beam size : {}
        max decoding length: {}
        *********************************
        '''.format(
            self.num_g, self.emb_dim, self.emb_dim, self.hidden_dim,
            self.num_p, self.emb_dim, self.emb_dim, self.hidden_dim,
            self.device, self.beam_size, self.max_decode_len)

    def forward(self, g_seq, p_seq=None):
        '''
        Input arguments:
        * g_seq (torch.Tensor): A [bz, g_len] shaped tensor containing the
        grapheme index values for each sample in the batch where g_len is
        the padded batch
        length of the grapheme input
        * p_seq (None or torch.Tensor): A [bz, p_len] shaped tensor containing
        the phoneme index values for each sample in the batch where p_len is
        the padded batch length of the phoneme input
        '''

        context, h, c = self.encoder(g_seq)
        if p_seq is not None:
            # We are training
            return self.decoder(p_seq, h, c, context)
        else:
            # We are generating
            assert g_seq.size(1) == 1  # make sure batch_size = 1
            return self._generate(h, c, context)

    def _generate(self, h, c, context):
        '''
        Input arguments:
        TODO: ADD DOCUMENT
        '''
        beam = Beam(self.beam_size, device=self.device)
        # Make a beam_size batch.
        h = h.expand(beam.size, h.size(1))
        c = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))

        for _ in range(self.max_decode_len):
            x = beam.get_current_state()
            o, h, c = self.decoder(Variable(x.unsqueeze(0)), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h.data.copy_(h.data.index_select(0, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(0, beam.get_current_origin()))
        return torch.Tensor(beam.get_hyp(0)).to(
            dtype=torch.long, device=self.device)


class Encoder(nn.Module):
    def __init__(self, num_g: int, emb_dim: int, hidden_dim: int, device):
        '''
        Input arguments:
        * num_g (int): The size of the grapheme vocabulary
        * emb_dim (int): The dimensionality of the character embeddings used
        * hidden_dim (int): The hidden dimensionality of the encoder RNN
        * device (torch.device): The device to store this module
        '''
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_g, emb_dim)
        #  self.lstm = nn.LSTMCell(emb_dim, hidden_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

        self.device = device

    def forward_2(self, g_seq):
        '''
        Input arguments:
        * g_seq (torch.Tensor): A [bz, seq_g] shaped tensor containing the
        grapheme index values for each sample in the batch where seq_g is
        the padded batch length of the grapheme input
        '''
        o = []
        e_seq = self.embedding(g_seq)  # batch x seq_g x emb_dim

        # create initial hidden state and initial cell state
        h = torch.zeros((e_seq.shape[0], self.hidden_dim)).to(
            dtype=torch.float, device=self.device)  # bz x hidden_dim
        c = torch.zeros((e_seq.shape[0], self.hidden_dim)).to(
            dtype=torch.float, device=self.device)  # bz x hidden_dim

        for e in e_seq.chunk(e_seq.shape[1], 1):  # iterate sequence
            e = e.squeeze(1)  # bz x emb_dim
            h, c = self.lstm(e, (h, c))
            o.append(h)

        o = torch.stack(o, dim=1)  # bz x seq_g x hidden_dim
        return o, h, c

    def forward(self, g_seq):
        '''
        Input arguments:
        * g_seq (torch.Tensor): A (bz x seq_g) shaped tensor containing the
        grapheme index values for each sample in the batch where seq_g is
        the padded batch length of the grapheme input

        Returns:
        * out (torch.Tensor): A (bz x seq_g x hidden_dim) tensor containing
        the RNN output from all timesteps
        * h (torch.Tensor): A (bz x hidden_dim) tensor containing the last
        hidden state of the RNN
        * c (torch.Tensor): A (bz x hidden_dim) tensor containing the last
        cell state of the RNN
        '''
        g_seq = self.embedding(g_seq)  # batch x seq_g x emb_dim
        out, (h, c) = self.lstm(g_seq, self.init_hidden(g_seq.shape[0]))
        return out, h.squeeze(dim=0), c.squeeze(dim=0)

    def init_hidden(self, bz: int):
        '''
        Input arguments:
        * bz (int): The current batch size being used

        Returns: (h_0, c_0)
        * h_0 (torch.Tensor): A (bz x hidden_dim) tensor containing the
        initial hidden state
        * c_0 (torch.Tensor):  A (bz x hidden_dim) tensor containing the
        initial cell state
        '''
        return (
            torch.zeros((1, bz, self.hidden_dim)).to(
                dtype=torch.float, device=self.device),
            torch.zeros((1, bz, self.hidden_dim)).to(
                dtype=torch.float, device=self.device))


class Decoder(nn.Module):
    def __init__(self, num_p: int, emb_dim: int, hidden_dim: int):
        '''
        Input arguments:
        * num_p (int): The size of the phoneme vocabulary
        * emb_dim (int): The dimensionality of the character embeddings used
        * hidden_dim (int): The hidden dimensionality of the decoder RNN
        '''
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_p, emb_dim)
        self.lstm = nn.LSTMCell(emb_dim, hidden_dim)
        self.attn = Attention(hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_p)

    def forward(self, p_seq, h, c, context=None):
        '''
        Input arguments:
        * p_seq (torch.Tensor): A [bz, seq_p] shaped tensor containing the
        phoneme index values for each sample in the batch where p_seq is
        the padded batch length of the phoneme output
        * h: (torch.Tensor): A [bz, hidden_dim] tensor containing the last
        hidden emission from the encoder RNN
        * c: (torch.Tensor): A [bz, hidden_dim] tensor containing the last
        cell state for the encoder RNN
        * context: (torch.Tensor): A [bz, seq_g, hidden_dim] shaped tensor
        containing the hidden emissions from all encoder time steps

        Returns:
        * o (torch.Tensor): A (bz x seq_p, num_phonemes) tensor where
        o[i, j, k] is the probability that the j-th phoneme in the i-th
        batch is the k-th phoneme
        * h (torch.Tensor): A (bz x hidden_dim) tensor containing the last
        hidden emission of the decoder RNN
        * c (torch.Tensor): A (bz x hidden_dim) tensor containing the last cell
        state of the decoder RNN
        '''
        o = []
        e_seq = self.embedding(p_seq)  # bz x seq_p x hidden_dim
        for e in e_seq.chunk(e_seq.shape[1], 1):  # iterate sequence
            e = e.squeeze(1)
            h, c = self.lstm(e, (h, c))
            attn_i, ctxt_i = self.attn(h, context)
            o.append(ctxt_i)

        o = torch.stack(o, dim=1)
        o = self.linear(o.view(-1, h.size(1)))
        return (
            F.log_softmax(o, dim=1).view(-1, p_seq.shape[1], o.size(1)), h, c)


class Attention(nn.Module):
    def __init__(self, dim: int):
        '''
        Input arguments:
        * dim (int): The hidden dimensionality of the encoder and
        decoder RNNs
        '''
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim*2, dim, bias=False)

    def forward(self, h, context=None):
        '''
        Input arguments:
        * h (torch.Tensor): A (bz x hidden_dim) shaped hidden emission
        from the decoder
        * context (torch.Tensor) The (bz x seq_g, hidden_dim) shaped context
        of all hidden emissions from the encoder

        Returns
        * attn (torch.Tensor): A (bz x seq_g) tensor containing the attention
        score for the current decoder time step over all possible encoder time
        steps
        * context (torch.Tensor): A (bz x hidden_dim) tensor containing
        tanh(FC(h | context @ attn))
        '''

        if context is None:
            return None, h

        assert h.shape[0] == context.shape[0]
        assert h.shape[1] == context.shape[2]
        attn = F.softmax(context.bmm(h.unsqueeze(2)).squeeze(2), dim=1)
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)
        o = self.linear(torch.cat((h, weighted_context), 1))
        return attn, torch.tanh(o)


if __name__ == '__main__':
    from PronSet import g_vocab,  p_vocab, PronSet, extract_pron
    graphemes, phonemes = extract_pron('./data/prondict_ice.txt')
    ds = PronSet(graphemes, phonemes)
    idx, batch = next(enumerate(ds.get_loader()))

    model = G2P(len(g_vocab), len(p_vocab), 128, 256)

    model(batch[0], batch[1])