# -*- coding: utf-8 -*-
#
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

import operator
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from score import phonemes, sil_diphones, sil_phone, sil_phonemes


class PronData:
    def __init__(
            self, src_path: str, needed_dps=[], all_phones=sil_phonemes,
            all_diphones=sil_diphones, contains_scores=False, num_needed=20):
        '''
        A class for many different kinds of operations on the data schemas
        used in this work.

        Input arguments:
        * src_path (str): A path to a G2P output file where each line
        contains a token and a pronounciation
        * all_phones (list): A list of all phones.
        * all_diphones (list) : A list of all possible diphones where each
        item in the list is a concatenated string of two phonemes.
        * contains_scores (bool): If True, each line in the input file
        is e.g. <sentence>\t<source_id>\t<score> else it is
        <sentence>\t<source_id>
        * num_needed (int): This is relevant when finding dihpones missing in
        other PronData sets. It is important only to
        self.export_needed_from_ids() and self.collect_needed_diphones()
        '''
        self.contains_scores = contains_scores
        self.tokens, self.srcs, self.scrs, self.prons, self.diphones,\
            self.lines = [], [], [], [], [], []
        self.all_diphones = all_diphones
        self.all_phones = all_phones
        # the number of occurrences of all diphones
        self.phone_counts = {p: 0 for p in self.all_phones}
        self.diphone_counts = {dp: 0 for dp in self.all_diphones}
        # diphones encountered that are possibly not valid
        self.bad_diphones = defaultdict(int)
        # A phone-by-phone array with number of occurs on the 3rd axis
        self.diphone_map = defaultdict(lambda: defaultdict(int))
        self.num_nonzero = 0

        self.needed_ids = []

        with open(src_path, 'r') as g2p_file:
            for _, line in tqdm(enumerate(g2p_file)):
                self.lines.append(line)
                if contains_scores:
                    token, src, scr, *phone_strings = line.split('\t')[0:]
                    self.scrs.append(scr)
                else:
                    token, src, *phone_strings = line.split('\t')[0:]
                self.tokens.append(token)
                self.srcs.append(src)
                self.prons.append(phone_strings)
                phones = self.sentence_2_phones(phone_strings)
                for p in phones:
                    self.phone_counts[p] += 1
                diphones = self.sentence_2_diphones(phone_strings)
                valid = []
                utt_added = False
                for diphone in diphones:
                    try:
                        self.diphone_counts[self.dpkey(diphone)] += 1
                        self.diphone_map[diphone[0]][diphone[1]] += 1
                        valid.append(diphone)
                        if self.dpkey(diphone) in needed_dps:
                            needed_dps[self.dpkey(diphone)] += 1
                            if needed_dps[self.dpkey(diphone)] >= num_needed:
                                needed_dps.pop(self.dpkey(diphone))
                            # append this utt only once !
                            if not utt_added:
                                self.needed_ids.append(len(self.diphones))
                                utt_added = True
                    except KeyError:
                        self.bad_diphones[diphone] += 1
                self.diphones.append(valid)
        self.num_nonzero = sum([
            1 for key, val in self.diphone_counts.items() if val > 0])

    def __len__(self):
        '''
        Returns the number of samples in this set
        '''
        return len(self.tokens)

    def get_utt(self, i: int):
        '''
        Returns the i-th utterance
        '''
        return self.tokens[i]

    def get_src(self, i: int):
        '''
        Returns the i-th source id
        '''
        return self.srcs[i]

    def get_pron(self, i: int):
        '''
        Returns the i-th phonetic prediction
        '''
        return self.prons[i]

    def get_diphones(self, i: int):
        '''
        Returns the diphones for the i-th utterance
        '''
        return self.diphones[i]

    def get_line(self, i: int):
        '''
        Returns the complete i-th line from the input file
        '''
        return self.lines[i]

    def get_phone_counts(self):
        '''
        Returns the number of each phone in this set
        '''
        return self.phone_counts

    def get_sources(self):
        '''
        Returns a list of unique sources in this collection
        '''
        return list(set(self.srcs))

    def dpkey(self, diphone):
        '''
        The standard dictionary key of a diphone is the
        concatenation of the two phones.

        Input arguments:
        * diphone (iterable): contains the two phones.
        '''
        return "".join(diphone)

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

    def word_2_phones(self, phone_string: str):
        '''
        Returns the individual phones in a space seperated string
        of phones
        '''
        return phone_string.split()

    def sentence_2_phones(self, ph_strings: list):
        '''
        Given a list of phone strings, this will return the complete
        phonetic prediction as a single string.
        '''
        return self.word_2_phones(' '.join(ph_strings))

    def coverage(self, dp_dict=None, n_needed=1):
        '''
        Returns the ratio of the number of covered diphones
        to the number of total diphones

        Input arguments:
        * dp_dict (dict or None): A dictionary where each key
        is a dihpone and each value is the number of occurrences.
        If dp_dict is none, self.dihpone_counts is usd
        * n_needed (int=1): A user chosen value, determining how
        many occurrences of each diphone is needed to fullfill 100%
        coverage.
        '''
        if dp_dict is None:
            dp_dict = self.diphone_counts

        total = 0.0
        for dp in self.all_diphones:
            total += min(dp_dict[dp], n_needed)/n_needed
        return total/len(self.all_diphones)

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

    def plot_coverage(self, fname: str = 'diphone_coverage.png'):
        '''
        Create a simple pinplot showing the total number of
        occurrences of each diphone, descended order by
        frequency.

        Input arguments:
        * fname (str): The name of the file to store the plot
        '''
        plt.clf()
        plt.bar(
            range(len(self.diphone_counts)),
            sorted(list(self.diphone_counts.values()), reverse=True),
            align='center')
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
        plt.clf()
        m = np.zeros([len(self.all_phones), len(self.all_phones)])
        for p, other_p in self.diphone_map.items():
            for pp, num in other_p.items():
                m[self.all_phones.index(p),
                    self.all_phones.index(pp)] = num
        _, ax = plt.subplots(figsize=(60, 60))
        _ = ax.imshow(m)
        ax.tick_params(axis='both', which='major', labelsize=80)
        ax.tick_params(axis='both', which='minor', labelsize=80)
        ax.set_xticks(np.arange(len(self.all_phones)))
        ax.set_yticks(np.arange(len(self.all_phones)))
        ax.set_xticklabels(self.all_phones)
        ax.set_yticklabels(self.all_phones)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    def collect_needed_diphones(self, dps, out_file: str, num_needed=20):
        '''
        Given a list of diphone ids, search the diphones in this set for
        utterances that contain some of the needed dihpones.
        '''
        o_f = open(out_file, 'w')

        for dp, num in tqdm(dps.items()):
            # first check if dp exists in this set
            if self.diphone_counts[dp] != 0:

                # needed is the min of the number we actually want
                # and what is actually available.
                needed = min(num_needed - num, self.diphone_counts[dp])
                # then search
                for i in self.needed_ids:
                    if dp in [self.dpkey(d) for d in self.get_diphones(i)]:
                        # we found one
                        o_f.write(self.get_line(i))
                        needed -= 1
                        if needed == 0:
                            break

    def export_needed_from_ids(self, out_file):
        o_f = open(out_file, 'w')
        for i in self.needed_ids:
            pron = "\t".join(self.get_pron(i))
            o_f.write(f'{self.get_utt(i)}\t{self.get_src(i)}\t{pron}')

    def greedy_coverage(self, dps):
        '''
        Measure greedy coverage by counting the number of
        diphones in dps that occur at least once that also
        occur n self.diphone_counts.

        The coverage of each diphone is equal to
        count_bag / min(5, count_set) where count_bag is the
        number of this diphone in the added set and
        count_set is the number of  this diphone in the total
        corpus.
        '''

        total_count = 0
        for dp, count in dps.items():
            if count != 0:
                num_occurs = self.diphone_counts[dp]
                if num_occurs > 0:
                    total_count += min(5, count) / min(5, num_occurs)
        return total_count / self.num_nonzero

    def diphones_to_file(self, out_path: str):
        '''
        Export the diphone frequency dictionary to a
        file, sorted by frequency descending
        '''
        with open(out_path, 'w') as o_f:
            diphones = sorted(
                self.diphone_counts.items(),
                key=operator.itemgetter(1), reverse=True)
            for d in diphones:
                o_f.write(f'{d[0]}\t{d[1]}\n')

    def export_needed_diphones(self, out_path: str, threshold=5):
        needed = {
            k: v for k, v in self.diphone_counts.items() if v < threshold}
        with open(out_path, 'wb') as handle:
            pickle.dump(needed, handle, protocol=pickle.HIGHEST_PROTOCOL)
