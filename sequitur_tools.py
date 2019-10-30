# -*- coding: utf-8 -*-

import os
import re
import math
import sequitur
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from g2p import SequiturTool, Translator, loadG2PSample

SEQUITUR_MDL_PATH = 'pron_data/ipd_clean_slt2018.mdl'
OTHER_CHARS = 'abdefghijklmnoprstuvxyz'
ICE_CHARS = 'áéíóúýæöþð'
ALL_CHARS = OTHER_CHARS + ICE_CHARS
sub_pattern = re.compile(r'[^{}]'.format(ALL_CHARS))

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

options = Options(
    modelFile=os.getenv("G2P_MODEL", SEQUITUR_MDL_PATH))
translator = Translator(SequiturTool.procureModel(options, loadG2PSample))

def predict(words, translator=translator):
    '''
    TODO : simplify this
    Input arguments:
    * words: A list of strings

    Yields:
    [{"word": word_1, "results":results_1}, ...]
    where word_1 os the first word in words and results_1 is a
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

def get_phones(token):
    '''
    token (str): A string of words seperated by a space forming a
    small sentence.

    '''
    words = [normalize_word(w) for w in token.strip().split()]
    predictions = list(predict(words))
    phones = []
    for pred in predictions:
        phones.append(pred['results'][0]['pronunciation'])
    return phones

def normalize_word(word, sub_pattern=sub_pattern):
    word = word.lower()
    word = re.sub(sub_pattern, '', word)
    return word

def g2p_file(src_path:str, out_path:str, n_jobs:int=4):
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
    '''
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []

    with open(src_path, 'r') as token_file:
        for line in token_file:
            token, src = line.split('\t')
            futures.append([token, src, executor.submit(
                partial(get_phones, token))])

    with open(out_path, 'w') as out_file:
        results = [
            (future[0], future[1], future[2].result()) for future
            in tqdm(futures) if future[2].result() is not None]
        for res in results:
            out_file.write('{}\t{}\t{}\n'.format(res[0].strip(),\
                res[1].strip(), '\t'.join(res[2][:])))