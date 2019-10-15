# -*- coding: utf-8 -*-

import os
import re
import math
import sequitur

from g2p import SequiturTool, Translator, loadG2PSample

OTHER_CHARS = 'abdefghijklmnoprstuvxyz'
ICE_CHARS = 'áéíóúýæöþð'
ALL_CHARS = OTHER_CHARS + ICE_CHARS
sub_pattern = re.compile(r'[^{}]'.format(ALL_CHARS))

class Options(dict):
    def __init__(self, modelFile="data/ipd_clean_slt2018.mdl", encoding="UTF-8",
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
    modelFile=os.getenv("G2P_MODEL", "data/ipd_clean_slt2018.mdl"))
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
    return token, phones

def normalize_word(word, sub_pattern=sub_pattern):
    word = word.lower()
    word = re.sub(sub_pattern, '', word)
    return word