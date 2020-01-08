"""Postprocessing for the PyTorch G2P model"""
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


class Beam:
    """
    Ordered beam of candidate outputs.
    See https://github.com/MaximumEntropy/Seq2Seq-PyTorch/ for more
    """
    def __init__(
            self, size, pad: int = 1, bos: int = 2, eos: int = 3,
            device=torch.device('cpu')):
        '''
        Input arguments:
        * size (int): The beam size
        * pad (int): The index of the <pad> token in the phoneme vocabulary
        * bos (int): The index of the <bos> token in the phoneme vocabulary
        * eos (int): The index of the <bos> token in the phoneme vocabulary
        '''

        self.size = size
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.done = False

        self.device = device

        # The score for each translation on the beam.
        self.scores = torch.Tensor(size).to(
            device=self.device, dtype=torch.float).zero_()

        # The backpointers at each time-step.
        self.prev_kk = []

        # The outputs at each time-step.
        self.next_yy = [torch.Tensor(size).to(
            device=self.device, dtype=torch.long).fill_(self.pad)]
        self.next_yy[0][0] = self.bos

    def get_current_state(self):
        '''
        Get the outputs for the current timestep.

        Returns: A (size) shaped tensor
        '''
        return self.next_yy[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prev_kk[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_kk) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_kk.append(prev_k)
        self.next_yy.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_yy[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prev_kk), len(self.next_yy), len(self.attn))
        for j in range(len(self.prev_kk) - 1, -1, -1):
            hyp.append(self.next_yy[j + 1][k])
            k = self.prev_kk[j][k]
        return hyp[::-1]
