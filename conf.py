"""Wrapper for global parameters"""
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
from pathlib import Path

# Parameters
MAX_WORDS = 15
MIN_WORDS = 5
VARIANTS_NUMBER = 4
VARIANTS_MASS = 0.9
DIPHONES_NEEDED = 20


# Paths
ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = ROOT_DIR / 'pron_data'
SEQUITUR_MDL_PATH = DATA_DIR / 'ipd_clean_slt2018.mdl'
VALID_DIPHONES_PATH = DATA_DIR / 'diphones/complete_ipa.txt'
XSAMPA_DIPHONES_PATH = DATA_DIR / 'diphones/diphones_xsampa.txt'
IPA_XSAMPA_PATH = DATA_DIR / 'ipa_2_xsampa.txt'
PRONDICT_IPA_PATH = DATA_DIR / 'prondict_ice.txt'
BIN_LIST_SORTED_PATH = DATA_DIR / 'bin_list_sorted.txt'

# Graphemes
ICE_ALPHABET = 'aábdðeéfghiíjklmnoóprstuúvxyýþæö'
OTHER_CHARS = ' .,?!:„“-'

# Phonemes
SIL_PHONE = '~'
PHONEMES = [
    'a', 'r', 't', 's', 'n', 'ɪ', 'l', 'ʏ', 'k', 'm',
    'ð', 'ɛ', 'v', 'p', 'h', 'f', 'j', 'c', 'i', 'ɔ', 'r̥',
    'ei', 'ŋ', 'ɣ', 'ou', 'œ', 'ouː', 'au', 'ai', 'aː', 'auː',
    'iː', 'eiː', 'ɪː', 'ɛː', 'θ', 'l̥', 'tʰ', 'uː', 'aiː',
    'kʰ', 'u', 'ɔː', 'x', 'œː', 'œy', 'n̥', 'cʰ', 'œyː', 'pʰ',
    'ɲ', 'ʏː', 'ç', 'ŋ̊', 'm̥', 'ʏi', 'ɲ̊', 'ɔi']

DIPHONES = []
with open(VALID_DIPHONES_PATH) as i_f:
    for line in i_f:
        p1, p2 = line.split('\t')
        DIPHONES.append("".join([p1, p2.strip()]))

for phone in PHONEMES:
    DIPHONES.append("".join([phone, SIL_PHONE]))
    DIPHONES.append("".join([SIL_PHONE, phone]))
