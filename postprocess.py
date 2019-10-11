import torch

class Beam(object):
    """
    Ordered beam of candidate outputs.
    See https://github.com/MaximumEntropy/Seq2Seq-PyTorch/ for more
    """
    def __init__(self, size, pad :int=1, bos :int=2, eos :int=3, device=torch.device('cpu')):
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
        self.scores = torch.Tensor(size).to(device=self.device,
            dtype=torch.float).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [torch.Tensor(size).to(device=self.device,
            dtype=torch.long).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    def get_current_state(self):
        '''
        Get the outputs for the current timestep.

        Returns: A (size) shaped tensor
        '''
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]
