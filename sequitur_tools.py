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

class Sequitur:
    def __init__(self):
        self.options = Options(
            modelFile=os.getenv("G2P_MODEL", "data/ipd_clean_slt2018.mdl"))
        self.translator = Translator(
            SequiturTool.procureModel(self.options, loadG2PSample))

        self.sub_pattern = re.compile(r'[^{}]'.format(ALL_CHARS))

    def predict(self, words):
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
                n_best = self.translator.nBestInit(left)
                while (
                        total_posterior < self.options.variants_mass and
                        n_variants < self.options.variants_number):
                    try:
                        log_like, result = self.translator.nBestNext(n_best)
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

    def get_phones(self, token):
        '''
        token (str): A string of words seperated by a space forming a
        small sentence.

        '''
        words = [self.normalize_word(w) for w in token.split()]
        predictions = list(self.predict(words))
        phones = []
        for pred in predictions:
            phones.append(pred['results'][0]['pronunciation'])
        return phones

    def normalize_word(self, word):
        '''
        '''
        word = word.lower()
        word = re.sub(self.sub_pattern, '', word)
        return word


class Options(dict):
    def __init__(self, modelFile="data/ipd_clean_slt2018.mdl", encoding="UTF-8",
                 variants_number=4, variants_mass=0.9):
        super(Options, self).__init__(modelFile=modelFile, encoding=encoding,
                                      variants_number=variants_number,
                                      variants_mass=variants_mass)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value