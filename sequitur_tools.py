# -*- coding: utf-8 -*-

import os
import re
import math
import sequitur
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from g2p import SequiturTool, Translator, loadG2PSample
from conf import ICE_ALPHABET, SEQUITUR_MDL_PATH
sub_pattern = re.compile(r'[^{}]'.format(ICE_ALPHABET))

class Options(dict):
    def __init__(self, modelFile=SEQUITUR_MDL_PATH, encoding="UTF-8",
                 variants_number=4, variants_mass=0.9):
        super(Options, self).__init__(
            modelFile=modelFile, encoding=encoding,
            variants_number=variants_number,
            variants_mass=variants_mass)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value

def predict(words, translator):
    '''
    Input arguments:
    * words (list): A list of strings
    * translator (g2p.Translator instance)

    Yields:
    [{"word": word_1, "results":results_1}, ...]
    where word_1 is the first word in words and results_1 is a
    list of dictionaries where e.g.
    result_1 = [{'posterior':posterior_1, 'pronounciation':pron_1}, ...]
    where pron_1 is a string of phoneme predictions, each phoneme seperated
    by a space.
    '''

    for word in words:
        left = tuple(word)
        output = {
            "word": word,
            "results": []}
        try:
            total_posterior = 0.0
            n_variants = 0
            n_best = translator.nBestInit(left)
            while (
                    total_posterior < options.variants_mass and
                    n_variants < options.variants_number):
                try:
                    log_like, result = translator.nBestNext(n_best)
                except StopIteration:
                    break
                posterior = math.exp(log_like - n_best.logLikTotal)
                output["results"].append(
                    {"posterior": posterior, "pronunciation": " ".join(
                        result)}
                )
                total_posterior += posterior
                n_variants += 1

        except Translator.TranslationFailure:
            pass
        yield output

def get_phones(token, translator):
    '''
    Takes a string forming a sentence and returns a list of
    phonetic predictions for each word.

    Input arguments:
    * token (str): A string of words seperated by a space forming a sentence.
    * translator (g2p.Translator instance)
    '''

    words = [normalize_word(w) for w in token.strip().split()]
    predictions = list(predict(words, translator=translator))
    phones = []
    for pred in predictions:
        phones.append(pred['results'][0]['pronunciation'])
    return phones

def normalize_word(word, sub_pattern=sub_pattern):
    '''
    A normalization step used specifically for Sequitur.
    The word given as input is lowercased and any character
    not matched by sub_pattern is replaced with the empty string.

    Input arguments:
    * word (string): The word to be normalized
    * sub_pattern (A compiled regex pattern): A substitution pattern
    '''
    word = word.lower()
    word = re.sub(sub_pattern, '', word)
    return word

def g2p_file(src_path:str, out_path:str, n_jobs:int=16, contains_scores=False,
    translator=None):
    '''
    Do grapheme-to-phoneme predictions on a list of tokens
    in a single file.

    Input arguments:
    * src_path (str): The path to the file containing multiple
    tokens, one per line
    * out_path (str): The target path the the file that stores
    the results.
    * n_jobs (int): The maximum number of processes that can
    be used to execute the given calls.
    * contains_scores (bool): If True, each line in the input file
    is e.g. <sentence>\t<source_id>\t<score> else it is
    <sentence>\t<source_id>
    * translator (g2p.Translator instance)
    '''
    if translator is None:
        options = Options(
            modelFile=os.getenv("G2P_MODEL", SEQUITUR_MDL_PATH))
        translator = Translator(SequiturTool.procureModel(options, loadG2PSample))

    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []

    if contains_scores:
        with open(src_path, 'r') as token_file:
            for line in token_file:
                token, src, scr = line.split('\t')
                futures.append([token, src, scr, executor.submit(
                    partial(get_phones, token, translator=translator))])

        with open(out_path, 'w') as out_file:
            results = [
                (future[0], future[1], future[2], future[3].result()) for future
                in tqdm(futures) if future[3].result() is not None]
            for res in results:
                out_file.write('{}\t{}\t{}\t~ {} ~\n'.format(res[0].strip(),\
                    res[1].strip(), res[2].strip(), '\t'.join(res[3][:])))
    else:
        with open(src_path, 'r') as token_file:
            for line in token_file:
                token, src = line.split('\t')
                futures.append([token, src, executor.submit(
                    partial(get_phones, token, translator=translator))])

        with open(out_path, 'w') as out_file:
            results = [
                (future[0], future[1], future[2].result()) for future
                in tqdm(futures) if future[2].result() is not None]
            for res in results:
                out_file.write('{}\t{}\t~ {} ~\n'.format(res[0].strip(),\
                    res[1].strip(), '\t'.join(res[2][:])))

if __name__ == '__main__':
    g2p_file('final_results/2.6.txt',
        'final_results/2.6_g2p.txt', contains_scores=False)