"""A parametric grapheme-to-phoneme model implemented in PyTorch"""
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


import torch
import torch.nn as nn
import torch.nn.functional as F

from postprocess import Beam


class G2P(nn.Module):
    """A PyTorch G2P module"""
    def __init__(
            self, num_g: int, num_p: int,
            emb_dim: int = 256, hidden_dim: int = 256,
            beam_size: int = 2,
            max_decode_len: int = 20):
        '''
        Input arguments:
        * num_g (int): The size of the grapheme vocabulary
        * num_p (int): The size of the phoneme voccabulary
        * emb_dim (int): The dimensionality of the character embeddings used
        * hidden_dim (int): The hidden dimensionality of the RNNs
        * beam_size (int=2): The size of decoding beams
        * max_decode_len (int=20): The maximum decoding output length
        '''
        super(G2P, self).__init__()
        self.encoder = Encoder(num_g, emb_dim, hidden_dim)
        self.decoder = Decoder(num_p, emb_dim, hidden_dim)

        self.num_g = num_g
        self.num_p = num_p
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.beam_size = beam_size
        self.max_decode_len = max_decode_len

    def summary(self) -> str:
        """Returns module summary"""
        return '''
        *********************************
        G2P model summary

        Grapheme Embedding : in/out {}/{}
        Encoder RNN : in/hidden {}/{}
        Phoneme Embedding : in/out {}/{}
        Decoder RNN : in/hidden {}/{}

        Is cuda : {}
        beam size : {}
        max decoding length: {}
        *********************************
        '''.format(
            self.num_g, self.emb_dim, self.emb_dim, self.hidden_dim,
            self.num_p, self.emb_dim, self.emb_dim, self.hidden_dim,
            next(self.parameters()).is_cuda, self.beam_size,
            self.max_decode_len)

    def forward(self, g_seq, p_seq=None, device=torch.device('cpu')):
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
        context, hidden, cell_state = self.encoder(g_seq, device=device)
        if p_seq is not None:
            # We are training
            return self.decoder(p_seq, hidden, cell_state, context)
        # We are generating
        assert g_seq.shape[0] == 1, "batch size must be one when testing"
        return self._gen(hidden, cell_state, context, device=device)

    def _gen(self, hidden, cell_state, context, device=torch.device('cpu')):
        '''
        Input arguments:
        * hidden (tensor): A (1 x hidden_dim) shaped tensor containinig the last
        hidden emission from the encoder
        * cell_state (tensor): A (1 x hidden_dim) shaped tensor containing the last
        cell state from the encoder
        * context (tensor) A (1 x seq_g x hidden_dim) shaped tensor containing
        encoder
        emissions for all encoder timesteps
        '''
        beam = Beam(self.beam_size, device=device)
        hidden = hidden.expand(beam.size, hidden.shape[1])  # (beam_sz x hidden_dim)
        cell_state = cell_state.expand(beam.size, cell_state.shape[1])  # (beam_sz x hidden_dim)
        # (beam_sz x seq_g x hidden_dim)
        context = context.expand(beam.size, context.shape[1], context.shape[2])

        for _ in range(self.max_decode_len):
            current_out = beam.get_current_state()  # (beam_size)
            out, hidden, cell_state = self.decoder(
                current_out.unsqueeze(1), hidden, cell_state, context)
            if beam.advance(out.data.squeeze(1)):  # (beam_size x num_phones)
                break
            hidden.data.copy_(hidden.data.index_select(0, beam.get_current_origin()))
            cell_state.data.copy_(cell_state.data.index_select(0, beam.get_current_origin()))
        return torch.Tensor(beam.get_hyp(0)).to(
            dtype=torch.long, device=device)


class Encoder(nn.Module):
    """PyTorch LSTM encoder module"""
    def __init__(self, num_g: int, emb_dim: int, hidden_dim: int):
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

    def forward_2(self, g_seq):
        '''
        Input arguments:
        * g_seq (torch.Tensor): A [bz, seq_g] shaped tensor containing the
        grapheme index values for each sample in the batch where seq_g is
        the padded batch length of the grapheme input
        '''
        out = []
        e_seq = self.embedding(g_seq)  # batch x seq_g x emb_dim

        # create initial hidden state and initial cell state
        hidden = torch.zeros((e_seq.shape[0], self.hidden_dim)).to(
            dtype=torch.float, device=self.device)  # bz x hidden_dim
        cell_state = torch.zeros((e_seq.shape[0], self.hidden_dim)).to(
            dtype=torch.float, device=self.device)  # bz x hidden_dim

        for emb in e_seq.chunk(e_seq.shape[1], 1):  # iterate sequence
            emb = emb.squeeze(1)  # bz x emb_dim
            hidden, cell_state = self.lstm(emb, (hidden, cell_state))
            out.append(hidden)

        out = torch.stack(out, dim=1)  # bz x seq_g x hidden_dim
        return out, hidden, cell_state

    def forward(self, g_seq, device=torch.device('cpu')):
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
        out, (hidden, cell_state) = self.lstm(g_seq, self.init_hidden(
            g_seq.shape[0], device=device))
        return out, hidden.squeeze(dim=0), cell_state.squeeze(dim=0)

    def init_hidden(self, batch_size: int, device=torch.device('cpu')):
        '''
        Input arguments:
        * batch_size (int): The current batch size being used

        Returns: (h_0, c_0)
        * h_0 (torch.Tensor): A (bz x hidden_dim) tensor containing the
        initial hidden state
        * c_0 (torch.Tensor):  A (bz x hidden_dim) tensor containing the
        initial cell state
        '''
        return (
            torch.zeros((1, batch_size, self.hidden_dim)).to(
                dtype=torch.float, device=device),
            torch.zeros((1, batch_size, self.hidden_dim)).to(
                dtype=torch.float, device=device))


class Decoder(nn.Module):
    """PyTorch decoder LSTM module with attention"""
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

    def forward(self, p_seq, hidden, cell_state, context=None):
        '''
        Input arguments:
        * p_seq (torch.Tensor): A [bz, seq_p] shaped tensor containing the
        phoneme index values for each sample in the batch where p_seq is
        the padded batch length of the phoneme output
        * hidden: (torch.Tensor): A [bz, hidden_dim] tensor containing the last
        hidden emission from the encoder RNN
        * cell_state: (torch.Tensor): A [bz, hidden_dim] tensor containing the last
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
        out = []
        e_seq = self.embedding(p_seq)  # bz x seq_p x hidden_dim
        for emb in e_seq.chunk(e_seq.shape[1], 1):  # iterate sequence
            emb = emb.squeeze(1)
            hidden, cell_state = self.lstm(emb, (hidden, cell_state))
            _, ctxt_i = self.attn(hidden, context)
            out.append(ctxt_i)

        out = torch.stack(out, dim=1)
        out = self.linear(out.view(-1, hidden.size(1)))
        return (
            F.log_softmax(out, dim=1).view(
                -1, p_seq.shape[1], out.size(1)), hidden, cell_state)


class Attention(nn.Module):
    """PyTorch attention module"""
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
        out = self.linear(torch.cat((h, weighted_context), 1))
        return attn, torch.tanh(out)
