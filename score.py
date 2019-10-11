# -*- coding: utf-8 -*-

import itertools
from collections import defaultdict

import sequitur_tools as seqtools

phonemes = [
    'a', 'r', 't', 's', 'n', 'ɪ', 'l', 'ʏ', 'k', 'm',
    'ð', 'ɛ', 'v', 'p', 'h', 'f', 'j', 'c', 'i', 'ɔ', 'r̥',
    'ei', 'ŋ', 'ɣ', 'ou', 'œ', 'ouː', 'au', 'ai', 'aː', 'auː',
    'iː', 'eiː', 'ɪː', 'ɛː', 'θ', 'l̥', 'tʰ', 'uː', 'aiː',
    'kʰ', 'u', 'ɔː', 'x', 'œː', 'œy', 'n̥', 'cʰ', 'œyː', 'pʰ',
    'ɲ', 'ʏː', 'ç', 'ŋ̊', 'm̥', 'ʏi', 'ɲ̊', 'ɔi']

diphones = itertools.combinations(phonemes, 2)

added_phones = defaultdict(int)

def get_diphones(token):
    words = token.split()
    diphones = []
    predictions = list(seqtools.predict(words))

    for pred in predictions:
        phones = pred['results'][0]['pronunciation'].split()
        diphones += [phones[i]+phones[i+1] for i in range(len(phones) - 1)]

    return diphones

def score(dihpones, length):
    score = 0

    for d in dihpones:
        added_phones[d] += 1
        score -= added_phones[d]
        print(score)

    return  score/length
if __name__ == '__main__':

    diphones = get_diphones('halló ég er heima')
    print(diphones)
    print(score(diphones, len('halló ég er heima')))
    print(score(diphones, len('halló ég er heima')))
    print(score(diphones, len('halló ég er heima')))