
import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from score import diphones, phonemes


class PronData:
    def __init__(self, src_path: str, all_phones=phonemes,
        all_diphones=diphones, banned_sources=['gold-blog', 'gold-websites', 'ivona']):

        '''
        Input arguments:
        * src_path (str): A path to a G2P output file where each line
        contains a token and a pronounciation
        * all_diphones (list) : A list of all possible diphones where each
        item in the list is a concatenated string of two phonemes.
        '''
        self.tokens, self.srcs, self.prons, self.diphones = [], [], [], []
        self.all_diphones = all_diphones
        self.all_phones = all_phones
        self.banned_sources = banned_sources
        # the number of occurrences of all diphones
        self.diphone_counts = {dp:0 for dp in all_diphones}
        # diphones encountered that are possibly not valid
        self.bad_diphones = defaultdict(int)
        # A phone-by-phone array with number of occurs on the 3rd axis
        self.diphone_map = defaultdict(lambda: defaultdict(int))

        with open(src_path, 'r') as g2p_file:
            for idx, line in tqdm(enumerate(g2p_file)):
                token, src, *phone_strings = line.split('\t')[0:]
                if src not in self.banned_sources:
                    self.tokens.append(token)
                    self.srcs.append(src)
                    self.prons.append(phone_strings)

                    diphones = self.sentence_2_diphones(phone_strings)
                    valid = []
                    for diphone in diphones:
                        try:
                            self.diphone_counts[self.dpkey(diphone)] += 1
                            self.diphone_map[diphone[0]][diphone[1]] += 1
                            valid.append(diphone)
                        except KeyError:
                            self.bad_diphones[diphone] += 1
                    self.diphones.append(valid)

    def word_2_diphones(self, phone_string: str):
        '''
        A string of space seperated phones phonetically
        representing a single word, e.g. for the original word
        "mig" the phone_string is "m ɪː ɣ". This will then
        return the list [("m", "ɪː"), ("ɪː", "ɣ")]

        This function will return the empty list if the phone_string
        includes only a single phone.

        Input arguments:
        * phone_string (str): An IPA space seperated phone string
        for a single word.
        '''
        phones = phone_string.split()
        return [(phones[i], phones[i+1]) for i in range(len(phones) - 1)]

    def sentence_2_diphones(self, ph_strings: list):
        '''
        This achieves the same as word_2_diphones but on the
        token level, meaning that between-word-diphones are
        counted as well.

        Input arguments:
        * ph_strings: A list of IPA space seperated phone strings,
        each one corresponding to a single word.
        '''
        return self.word_2_diphones(' '.join(ph_strings))

    def coverage(self):
        '''
        Returns the ratio of the number of covered diphones
        to the number of total diphones
        '''
        return len([k for k, v in self.diphone_counts.items() if v > 0])\
            / len(self.all_diphones)

    def missing_diphones(self, pd_path=None):
        '''
        Returns a list of diphones not currently covered.
        If pd_path is a path to an IPA pronounciation dictionary
        this function will return two lists, first is a list of
        dihpones covered neither in this dataset nor the
        dictionary and the second is only covered in the dictionary

        Input arguments:
        * pd_path (None or str): Possibly a path to an IPA
        pronounciation dictionary
        '''
        missing = [k for k, v in self.diphone_counts.items() if v == 0]
        if pd_path is None:
            return missing
        else:
            pd_dps = defaultdict(bool)
            with open(pd_path) as i_f:
                for line in i_f:
                    _, phones = line.split('\t')
                    for dp in self.word_2_diphones(phones):
                        pd_dps["".join(dp)] = True

            pron_cov, non_cov = [], []
            for m_dp in missing:
                if pd_dps[m_dp]:
                    pron_cov.append(m_dp)
                else:
                    non_cov.append(m_dp)

            return non_cov, pron_cov

    def plot_coverage(self, fname:str='diphone_coverage.png'):
        '''
        Create a simple pinplot showing the total number of
        occurrences of each diphone, descended order by
        frequency.

        Input arguments:
        * fname (str): The name of the file to store the plot
        '''
        plt.bar(range(len(self.diphone_counts)),
            sorted(list(self.diphone_counts.values()), reverse=True), align='center')
        plt.savefig(fname)
        plt.show()

    def plot_diphone_heatmap(self, fname='diphone_heatmap.png'):
        '''
        Create a phone-by-phone heatmap showing the frequency
        of each phone in relation to all other phones in all
        the diphones in this dataset.

        Input arguments:
        * fname (str): The name of the file to store the plot
        '''
        m = np.zeros([len(self.all_phones), len(self.all_phones)])
        for p, other_p in self.diphone_map.items():
            for pp, num in other_p.items():
                m[self.all_phones.index(p), self.all_phones.index(pp)] = num
        fig, ax = plt.subplots(figsize=(60, 60))
        im = ax.imshow(m)
        ax.tick_params(axis='both', which='major', labelsize=80)
        ax.tick_params(axis='both', which='minor', labelsize=80)
        ax.set_xticks(np.arange(len(self.all_phones)))
        ax.set_yticks(np.arange(len(self.all_phones)))
        ax.set_xticklabels(self.all_phones)
        ax.set_yticklabels(self.all_phones)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    def get_simple_score(self, i:int):
        '''
        Returns s(utt[i]) = 1/len(utt[i]) * sum_j [1/f(di[j])]
        where f(di[j]) is the corpus frequency of the j-th diphone
        in utt[i]

        Input arguments:
        * i (int): The index of the utterance to score
        '''
        diphones = self.get_diphones(i)

        score = 0.0
        for diphone in diphones:
            score += 1.0/self.diphone_counts[self.dpkey(diphone)]
        score *= 1/len(self.get_utt(i))
        return score

    def simple_score_file(self, out_path='scores.txt'):
        scores = []
        for i in range(len(self)):
            scores.append([self.get_utt(i), self.get_simple_score(i)])
        scores = sorted(scores, key=lambda r: r[1], reverse=True)

        out_file = open(out_path, 'w')
        for res in scores:
            out_file.write('{}\t{}\n'.format(*res))
        out_file.close()

    def get_utt(self, i:int):
        return self.tokens[i]

    def get_src(self, i:int):
        return self.srcs[i]

    def get_pron(self, i:int):
        return self.prons[i]

    def get_diphones(self, i:int):
        return self.diphones[i]

    def dpkey(self, diphone):
        '''
        The standard dictionary key of a diphone is the
        sequential concatenation of the two phones.

        Input arguments:
        * diphone (iterable): contains the two phones.
        '''
        return  "".join(diphone)

    def __len__(self):
        return len(self.tokens)


if __name__ == '__main__':
    p = PronData('./pron_data/no_repeats.txt')
    print(p.simple_score_file())