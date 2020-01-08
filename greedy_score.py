"""A greedy algorithm that scores diphone dense highly in
a large text corpus """
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

from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

from conf import DIPHONES


def init(g2p_file, contains_scores=False):
    """An initialization step for the greedy algorithm"""
    tokens, srcs, prons, diphones = [], [], [], []
    total_diphone_counts = {dp: 0 for dp in all_diphones}
    num_nonzero = 0

    with open(g2p_file, 'r') as g2p_file:
        for _, line in tqdm(enumerate(g2p_file)):
            if contains_scores:
                token, src, _, *phone_strings = line.split('\t')[0:]
            else:
                token, src, *phone_strings = line.split('\t')[0:]
            tokens.append(token)
            srcs.append(src)
            prons.append(phone_strings)

            new_diphones = sentence_2_diphones(phone_strings)
            valid = []
            for diphone in new_diphones:
                try:
                    total_diphone_counts[dpkey(diphone)] += 1
                    valid.append(diphone)
                except KeyError:
                    print(diphone)
                    pass
            diphones.append(valid)

    num_nonzero = sum(
        [1 for key, val in total_diphone_counts.items() if val > 0])
    return tokens, srcs, prons, diphones, total_diphone_counts, num_nonzero


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
    token level, meaning that between-word-diphones are
    counted as well.

    Input arguments:
    * ph_strings: A list of IPA space seperated phone strings,
    each one corresponding to a single word.
    '''
    return word_2_diphones(' '.join(ph_strings))


def get_utt(i: int):
    return tokens[i]


def get_src(i: int):
    return srcs[i]


def get_pron(i: int):
    return prons[i]


def get_diphones(i: int):
    return diphones[i]


def dpkey(diphone):
    '''
        The standard dictionary key of a diphone is the
    sequential concatenation of the two phones.

    Input arguments:
    * diphone (iterable): contains the two phones.
    '''
    return "".join(diphone)


def score(i: int):
    '''
    Returns s(utt[i]) = 1/len(utt[i]) * sum_j [1/f(di[j])]
    where f(di[j]) is the corpus frequency of the j-th diphone
    in utt[i]

    Input arguments:
    * i (int): The index of the utterance to score
    '''
    diphones = get_diphones(i)

    score = 0.0
    for diphone in diphones:
        score += 1.0/total_diphone_counts[dpkey(diphone)]
    score *= 1/len(get_utt(i))
    return score


def score_list(diphones, utt_len):
    '''
    Achieves the same as above but on a list of diphones
    '''
    score = 0.0
    for diphone in diphones:
        score += 1.0/total_diphone_counts[dpkey(diphone)]
    score *= 1/utt_len
    return score


def score_dict(diphones, utt_len):
    score = 0.0
    for diphone_key, _ in diphones.items():
        score += 1.0/total_diphone_counts[diphone_key]
    score *= 1/utt_len
    return score


def coverage(dp_dict=None):
    '''
    Returns the ratio of the number of covered diphones
    to the number of total diphones
    '''
    if dp_dict is None:
        dp_dict = total_diphone_counts
    return len([k for k, v in dp_dict.items() if v > 0])\
        / len(all_diphones)


def greedy_coverage(dps):
    '''
    Measure greedy coverage by counting the number of non-zero
    diphones in dps that are also non-zero in self.diphone_counts.

    The coverage of each diphone is equal to count_bag / min(5, count_set)
    where count_bag is the number of this diphone in the added set and
    count_set is the number of this diphone in the total corpus.
    '''

    total_count = 0
    for dp, count in dps.items():
        if count != 0:
            num_occurs = total_diphone_counts[dp]
            if num_occurs > 0:
                total_count += min(5, count) / min(5, num_occurs)
    return total_count / num_nonzero


def greedy_score_file(
        out_path='greedy_scores.txt', covplot_path='coverage_hist.png',
        added_dp_counts=None, num_needed=20, score_all=True):
    # TODO: add doc
    o_f = open(out_path, 'w')

    added, scores, unchanged, covs, greedy_covs = [], [], [], [], []
    diphone_counts = {dp: 0 for dp in all_diphones}

    if added_dp_counts is not None:
        for dp, num in added_dp_counts.items():
            diphone_counts[dp] += num
            total_diphone_counts[dp] += num
    not_needed = {}
    for dp, num in diphone_counts.items():
        if num >= num_needed:
            not_needed[dp] = True
    for i in range(len(tokens)):
        scores.append({
            'utt': get_utt(i),
            'src': get_src(i),
            'pron': get_pron(i),
            'dp_table': defaultdict(int),
            'scr': -1})
        for dp in get_diphones(i):
            if dpkey(dp) not in not_needed:
                scores[-1]['dp_table'][dpkey(dp)] += 1

    greedy_covs.append(greedy_coverage(diphone_counts))
    pbar = tqdm(range(len(tokens)))
    for _ in pbar:
        pbar.set_description(f"{greedy_covs[-1]}")
        # calculate the scores
        best, best_i = -1, -1
        for i, val in enumerate(scores):
            score_i = score_dict(val['dp_table'], len(val['utt']))
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
        for dp, num in new['dp_table'].items():
                diphone_counts[dp] += num
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
            for dp in dps_to_remove:
                scores[j]['dp_table'].pop(dp, None)
            if len(scores[j]['dp_table']) == len(dp_table):
                unchanged.append(scores.pop(j))
            j += 1

        covs.append(coverage(diphone_counts))
        greedy_covs.append(greedy_coverage(diphone_counts))
        if not score_all and greedy_covs[-1] >= 1:
            print("Reached full greedy coverage, quitting")
            break
    plt.plot(covs, c='b')
    plt.plot(greedy_covs, c='r')
    plt.show()
    plt.savefig(covplot_path)


if __name__ == '__main__':
    all_diphones = DIPHONES
    tokens, srcs, prons, diphones, total_diphone_counts, num_nonzero = init(
        'tests/unscored_list.txt', contains_scores=False)
    greedy_score_file('test.txt', 'test.png')
