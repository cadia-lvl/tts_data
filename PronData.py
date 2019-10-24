
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from score import diphones, phonemes

class PronData:
    def __init__(self, src_path: str, all_phones=phonemes,
        all_diphones=diphones):

        '''
        Input arguments:
        * src_path (str): A path to a G2P output file where each line
        contains a token and a pronounciation
        * all_diphones (list) : A list of all possible diphones where each
        item in the list is a concatenated string of two phonemes.
        '''
        self.tokens, self.srcs, self.prons = [], [], []
        self.all_diphones = all_diphones
        self.all_phones = all_phones
        self.diphones = {dp:0 for dp in all_diphones}
        self.bad_diphones = defaultdict(int)
        self.diphone_map = defaultdict(lambda: defaultdict(int))

        with open(src_path, 'r') as g2p_file:
            for idx, line in tqdm(enumerate(g2p_file)):
                token, src, *phone_strings = line.split('\t')[0:]
                self.tokens.append(token)
                self.srcs.append(src)
                self.prons.append(phone_strings)
                for phone_string in phone_strings:
                    diphones = self.get_diphones(phone_string)
                    for diphone in diphones:
                        try:
                            self.diphones["".join(diphone)] += 1
                            self.diphone_map[diphone[0]][diphone[1]] += 1
                        except KeyError:
                            self.bad_diphones[diphone] += 1

    def get_diphones(self, phone_string: str):
        '''
        A string of space seperated phones phonetically
        representing a single word, e.g. for the original word
        "mig" we get phone_string="m ɪː ɣ"
        '''
        phones = phone_string.split()
        return [(phones[i], phones[i+1]) for i in range(len(phones) - 1)]

    def coverage(self):
        '''
        Returns the ratio of the number of covered diphones
        to the number of total diphones
        '''
        return len([k for k, v in self.diphones.items() if v > 0])\
            / len(self.all_diphones)

    def missing_diphones(self):
        '''
        Returns a list of diphones not currently coveraged
        '''
        return [k for k, v in self.diphones.items() if v == 0]

    def plot_coverage(self, fname:str='diphone_coverage.png'):
        '''
        Create a simple pinplot showing the total number of
        occurrences of each diphone, descended order by
        frequency.

        Input arguments:
        * fname (str): The name of the file to store the plot
        '''
        plt.bar(range(len(self.diphones)),
            sorted(list(self.diphones.values()), reverse=True), align='center')
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

if __name__ == '__main__':
    P = PronData('data/processed_tokens/all_tokens_norm_clean_g2p.txt')
    print(P.coverage())
    P.plot_coverage('eiki_coverage.png')
    P.plot_diphone_heatmap('eiki_diphone_heatmap.png')