"""A module for many different kinds of operations on the data schemas
used in this work."""
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

import os
import operator
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from conf import DIPHONES, PHONEMES, SIL_PHONE, DIPHONES_NEEDED


class PronData:
    """
    A class for many different kinds of operations on the data schemas
    used in this work.
    """
    def __init__(
            self, src_path: str, needed_dps: dict = None,
            all_phones: list = PHONEMES+[SIL_PHONE],
            all_diphones: list = DIPHONES, contains_scores: bool = False,
            num_needed: int = DIPHONES_NEEDED):
        '''
        Input arguments:
        * src_path (str): A path to a G2P output file where each line
        contains an utterance and a pronounciation
        * needed_dps = (list/None = None): A list of diphones that are
        needed that potentially exist in this dataset. This is only relevant
        for self.export_needed_from_ids() and self.collect_needed_diphones()
        * all_phones (list): A list of all phones.
        * all_diphones (list) : A list of all possible diphones where each
        item in the list is a concatenated string of two phonemes.
        * contains_scores (bool): If True, each line in the input file
        is e.g. <sentence>\t<source_id>\t<score> else it is
        <sentence>\t<source_id>
        * num_needed (int = conf.DIPHONES_NEEDED): This is relevant when
        finding dihpones missing in other PronData sets. It is important only to
        self.export_needed_from_ids() and self.collect_needed_diphones()
        '''
        self.name = os.path.splitext(os.path.basename(src_path))[0]
        self.contains_scores = contains_scores
        self.utts, self.srcs, self.scrs, self.prons, self.diphones,\
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
                    utt, src, scr, *phone_strings = line.split('\t')[0:]
                    self.scrs.append(scr)
                else:
                    utt, src, *phone_strings = line.split('\t')[0:]
                self.utts.append(utt)
                self.srcs.append(src)
                self.prons.append(phone_strings)
                phones = self.sentence_2_phones(phone_strings)
                for phone in phones:
                    self.phone_counts[phone] += 1
                diphones = self.sentence_2_diphones(phone_strings)
                valid = []
                utt_added = False
                for diphone in diphones:
                    try:
                        self.diphone_counts[self.dpkey(diphone)] += 1
                        self.diphone_map[diphone[0]][diphone[1]] += 1
                        valid.append(diphone)
                        if needed_dps is not None and self.dpkey(diphone) in needed_dps:
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
        return len(self.utts)

    def get_utt(self, i: int):
        '''
        Returns the i-th utterance
        '''
        return self.utts[i]

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
        utterance level, meaning that between-word-diphones are
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

    def coverage(self, dp_dict=None, num_needed=DIPHONES_NEEDED):
        '''
        Returns the ratio of the number of covered diphones
        to the number of total diphones

        Input arguments:
        * dp_dict (dict or None): A dictionary where each key
        is a dihpone and each value is the number of occurrences.
        If dp_dict is none, self.dihpone_counts is usd
        * num_needed (int=conf.DIPHONES_NEEDED): A user chosen value,
        determining how many occurrences of each diphone is needed to
        fullfill 100% coverage.
        '''
        if dp_dict is None:
            dp_dict = self.diphone_counts

        total = 0.0
        for diphone in self.all_diphones:
            total += min(dp_dict[diphone], num_needed)/num_needed
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
                for diphone in self.word_2_diphones(phones):
                    pd_dps["".join(diphone)] = True

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
        phone_map = np.zeros([len(self.all_phones), len(self.all_phones)])
        for phone, others in self.diphone_map.items():
            for other_phone, num in others.items():
                phone_map[
                    self.all_phones.index(phone),
                    self.all_phones.index(other_phone)] = num
        _, axis = plt.subplots(figsize=(60, 60))
        _ = axis.imshow(phone_map)
        axis.tick_params(axisis='both', which='major', labelsize=80)
        axis.tick_params(axisis='both', which='minor', labelsize=80)
        axis.set_xticks(np.arange(len(self.all_phones)))
        axis.set_yticks(np.arange(len(self.all_phones)))
        axis.set_xticklabels(self.all_phones)
        axis.set_yticklabels(self.all_phones)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    def collect_needed_diphones(self, dps, out_file: str, num_needed=20):
        '''
        Given a list of diphone ids, search the diphones in this set for
        utterances that contain some of the needed dihpones.
        '''
        o_f = open(out_file, 'w')

        for diphone, num in tqdm(dps.items()):
            # first check if dp exists in this set
            if self.diphone_counts[diphone] != 0:

                # needed is the min of the number we actually want
                # and what is actually available.
                needed = min(num_needed - num, self.diphone_counts[diphone])
                # then search
                for i in self.needed_ids:
                    if diphone in [self.dpkey(d) for d in self.get_diphones(i)]:
                        # we found one
                        o_f.write(self.get_line(i))
                        needed -= 1
                        if needed == 0:
                            break

    def export_needed_from_ids(self, out_file: str):
        """Exports the lines from the input file that include diphones that
        are needed in this dataset"""
        o_f = open(out_file, 'w')
        for i in self.needed_ids:
            pron = "\t".join(self.get_pron(i))
            o_f.write(f'{self.get_utt(i)}\t{self.get_src(i)}\t{pron}')

    def greedy_coverage(self, dps, num_needed: int = DIPHONES_NEEDED):
        '''
        Measure greedy coverage by counting the number of
        diphones in dps that occur at least once that also
        occur n self.diphone_counts.

        The coverage of each diphone is equal to
        count_bag / min(num_needed, count_set) where count_bag is the
        number of this diphone in the added set and
        count_set is the number of  this diphone in the total
        corpus.
        '''
        total_count = 0
        for diphone, count in dps.items():
            if count != 0:
                num_occurs = self.diphone_counts[diphone]
                if num_occurs > 0:
                    total_count += min(num_needed, count) / min(num_needed, num_occurs)
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
            for diphone in diphones:
                o_f.write(f'{diphone[0]}\t{diphone[1]}\n')

    def export_needed_diphones(self, out_path: str, num_needed=5):
        """Pickles the diphones and how often they appear that are needed
        in this dataset"""
        needed = {
            k: v for k, v in self.diphone_counts.items() if v < num_needed}
        with open(out_path, 'wb') as handle:
            pickle.dump(needed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def score_dict(self, diphones: dict, utt_len: int):
        """
        A convenience method used in self.score

        Input arguments:
        * diphones (dict): A dictionary where each key is
        a diphone key as defined by self.dpkey()
        * utt_len (int): The length in characters of the sentence
        currently being considered.
        """
        score = 0.0
        for diphone_key, _ in diphones.items():
            score += 1.0/self.diphone_counts[diphone_key]
        score *= 1/utt_len
        return score

    def score(
            self, out_path: str, added_dp_counts: dict = None,
            num_needed: int = DIPHONES_NEEDED, score_all: bool = True,
            covplot_path: str = None):
        '''
        Greedy scoring algorithm to maximize total diphone coverage
        while minimizing sentence length.

        Input arguments:
        * out_path (str): Path for the output file generated by this method
        * added_dp_counts: (dict/None): If appending to another already scorede
        list, the diphone frequencies can be imported here as a {diphone: #}
        dictionary.
        * num_needed (int=conf.DIPHONES_NEEDED): The number needed of each
        diphone type to qualify for 100% coverage
        * score_all (bool=True): If False, the algorithm quits when maximum
        coverage has been reached w.r.t. the num_needed argument.
        * covplot_path (str/None): The path for the coverage plot generated
        by this method. If None, this defaults to <self.name>.png
        '''

        o_f = open(out_path, 'w')

        added, scores, unchanged, covs, greedy_covs = [], [], [], [], []
        diphone_counts = {dp: 0 for dp in self.all_diphones}

        if added_dp_counts is not None:
            for diphone, num in added_dp_counts.items():
                diphone_counts[diphone] += num
                self.diphone_counts[diphone] += num
        not_needed = {}
        for diphone, num in diphone_counts.items():
            if num >= num_needed:
                not_needed[diphone] = True
        for i in range(len(self)):
            scores.append({
                'utt': self.get_utt(i),
                'src': self.get_src(i),
                'pron': self.get_pron(i),
                'dp_table': defaultdict(int),
                'scr': -1})
            for diphone in self.get_diphones(i):
                if self.dpkey(diphone) not in not_needed:
                    scores[-1]['dp_table'][self.dpkey(diphone)] += 1

        greedy_covs.append(self.greedy_coverage(
            diphone_counts, num_needed=num_needed))
        pbar = tqdm(range(len(self)))
        for _ in pbar:
            pbar.set_description(f"{greedy_covs[-1]}")
            # calculate the scores
            best, best_i = -1, -1
            for i, val in enumerate(scores):
                score_i = self.score_dict(val['dp_table'], len(val['utt']))
                val['scr'] = score_i
                if score_i > best:
                    best, best_i = score_i, i
            for i, val in enumerate(unchanged):
                score_i = val['scr']
                if score_i > best:
                    best, best_i = score_i, i+len(scores)
            scores += unchanged
            unchanged = []

            # remove the best and add to the list
            new = scores.pop(best_i)
            for diphone, num in new['dp_table'].items():
                diphone_counts[diphone] += num
            added.append(new)
            o_f.write('{}\t{}\t{}\t{}\n'.format(
                added[-1]['utt'], added[-1]['src'],
                added[-1]['scr'], '\t'.join(p.strip() for p in added[-1]['pron'])))

            # remove the newly added diphones from all other elements
            dps_to_remove = [
                d for d in new['dp_table'] if
                diphone_counts[d] > min(num_needed, diphone_counts[d])]

            j = 0
            while j < len(scores):
                # only remove those diphones that occur min(5, num_in_corpus)
                # in the added set
                dp_table = scores[j]['dp_table']
                for diphone in dps_to_remove:
                    scores[j]['dp_table'].pop(diphone, None)
                if len(scores[j]['dp_table']) == len(dp_table):
                    unchanged.append(scores.pop(j))
                j += 1

            covs.append(self.coverage(diphone_counts, num_needed=num_needed))
            greedy_covs.append(self.greedy_coverage(
                diphone_counts, num_needed=num_needed))
            if not score_all and greedy_covs[-1] >= 1:
                print("Reached full greedy coverage, quitting")
                break
        plt.plot(covs, c='b')
        plt.plot(greedy_covs, c='r')
        plt.show()
        if covplot_path is None:
            covplot_path = f"{self.name}.png"
        plt.savefig(covplot_path)

def word_2_diphones(phone_string: str):
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

def sentence_2_diphones(ph_strings: list):
    '''
    This achieves the same as word_2_diphones but on the
    utterance level, meaning that between-word-diphones are
    counted as well.

    Input arguments:
    * ph_strings: A list of IPA space seperated phone strings,
    each one corresponding to a single word.
    '''
    return word_2_diphones(' '.join(ph_strings))

def dpkey(diphone):
    '''
    The standard dictionary key of a diphone is the
    concatenation of the two phones.

    Input arguments:
    * diphone (iterable): contains the two phones.
    '''
    return "".join(diphone)



class MultiSpeakerPron:
    def __init__(self, sorted_path:str, out_dir:str, num_speakers:int, num_same:int = 0,
        all_diphones: list = DIPHONES):
        '''
        Generate a number of reading lists from a pre-generated
        single speaker reading list. If num_same > 0 then the first
        num_same sentences will be in each reading list.

        Input arguments:
        * sorted_path (str): A path to a prondata .txt output file
        containing a sorted list of sentences
        * num_speakers (int): The number of speakers in the multi-speaker
        dataset
        * num_same (int): How many of the first sentences should be common
        for all speakers
        '''
        self.sorted_path = sorted_path
        self.out_dir = out_dir
        self.num_speakers = num_speakers
        self.num_same = num_same
        self.diphone_counts = {dp: 0 for dp in all_diphones}
        self.verbose = False

        # set up data
        self.utts = self.setup_utts()
        # set up speakers
        self.speakers = self.setup_speakers()
        # start algo
        self.add_utts_to_speakers()
        self.generate_scripts()

    def setup_utts(self):
        utts = []
        with open(self.sorted_path, 'r') as f:
            for _, line in tqdm(enumerate(f)):
                txt, src, scr, *phone_strings = line.strip().split('\t')[0:]
                diphones = sentence_2_diphones(phone_strings)
                utts.append(Utt(txt, phone_strings, src, scr))
                for diphone in diphones:
                    try:
                        self.diphone_counts[dpkey(diphone)] += 1
                        utts[-1].add_diphone(dpkey(diphone))
                    except KeyError:
                        if self.verbose:
                            print(f'{dpkey(diphone)} is an invalid diphone')
        return utts

    def setup_speakers(self):
        speakers = [Speaker() for _ in range(self.num_speakers)]
        # add the first num_same sentences to each
        for speaker in speakers:
            for i in range(self.num_same):
                speaker.add_utt(self.utts[i])

        return speakers

    def add_utts_to_speakers(self):
        while len(self.utts) > 0:
            best_score = -1
            best_speaker = None
            for idx, speaker in enumerate(self.speakers):
                score = speaker.score_utt(self.utts[0])
                if score > best_score:
                    best_score = score
                    best_speaker = speaker
                elif score == best_score and speaker.num_utts < best_speaker.num_utts:
                    best_speaker = speaker
            best_speaker.add_utt(self.utts[0])
            del self.utts[0]

        for idx, speaker in enumerate(self.speakers):
            print(f'{idx} - {speaker.num_utts}')

    def generate_scripts(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        for idx, speaker in enumerate(self.speakers):
            with open(os.path.join(self.out_dir, f'speaker_{idx+1}.txt'), 'w') as f:
                for utt in speaker.utts:
                    f.write(f'{utt.get_line()}\n')

class Utt:
    def __init__(self, text:str, phone_strings: str, src:str, scr:float):
        self.text = text
        self.phone_strings = phone_strings
        self.diphones = defaultdict(int)
        self.src = src
        self.scr = scr

    def add_diphone(self, dp):
        self.diphones[dp] += 1

    def get_line(self):
        return '\t'.join([self.text, self.src, self.scr, *self.phone_strings])

    def get_diphones(self):
        return self.diphones.items()

class Speaker:
    def __init__(self):
        self.diphones = defaultdict(int)
        self.utts = []

    def score_utt(self, utt):
        '''
        '''
        score = 0
        for diphone, num in utt.get_diphones():
            score += self.score_diphone(diphone, num)
        return score / self.num_utts**2

    def score_diphone(self, dp, num):
        return max(0, num - self.diphones[dp])**2

    def add_utt(self, utt):
        self.utts.append(utt)
        for diphone, num in utt.get_diphones():
            self.diphones[diphone] += num

    @property
    def num_utts(self):
        return len(self.utts)

def compare_scripts(in_dir):
    for f in os.listdir(in_dir):
        p = PronData(os.path.join(in_dir, f), contains_scores=True)
        print(p.coverage(num_needed=3))



if __name__ == '__main__':
    m = MultiSpeakerPron('./reading_lists/rl_full.txt', './multi/', 40, 100)
    #compare_scripts('./multi/')