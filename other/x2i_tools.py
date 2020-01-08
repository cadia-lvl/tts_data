"""XSAMPA to IPA converter and other related functions"""
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

from conf import XSAMPA_DIPHONES_PATH, IPA_XSAMPA_PATH


def x2i_diphones(
        out_path: str, i2x_path: str = IPA_XSAMPA_PATH,
        xdp_path: str = XSAMPA_DIPHONES_PATH, ipa_ind: int = 0,
        xsampa_ind: int = 2):
    '''
    Convert a list of XSAMPA diphones to a list of IPA diphones. Additionally
    if the symbol '_' appears in the diphone we skip the diphone.

    Input arguments:
    * out_path (str): The target path to store the IPA output in
    the same format
    * i2x_path (str): The path to the IPA XSAMPA index
    * ipa_ind (int): The index of the IPA symbol in each line
    of the source file when the line is split on \t
    * xsampa_ind (int): Same, but the index of the XSAMPA symbol
    xdp_path (str): The path to the list of XSAMPA diphones where
    each line is a tab seperated tuple.
    '''
    xs2ipa = {}
    with open(i2x_path) as m_f:
        for line in m_f:
            phones = line.split('\t')
            ipa_symbol, xsampa_symbol = phones[ipa_ind], phones[xsampa_ind]
            xs2ipa[xsampa_symbol.strip()] = ipa_symbol.strip()

    with open(xdp_path) as i_f, open(out_path, 'w') as o_f:
        for line in i_f:
            phone_1, phone_2 = line.split(' ')
            if '_' not in (phone_1.strip(), phone_2.strip()):
                o_f.write("{}\t{}\n".format(
                    xs2ipa[phone_1.strip()], xs2ipa[phone_2.strip()]))


def x2i_map(
        i2x_path: str = IPA_XSAMPA_PATH, ipa_ind: int = 0,
        xsampa_ind: int = 1) -> dict:
    '''
    Generate a mapping from XSAMPA to IPA in the
    form of a dictionary. This function expects a
    tab seperated file of IPA, XSAMPA symbols per line.

    Input arguments:
    * i2x_path (str): The path to the IPA XSAMPA index
    * ipa_ind (int): The index of the IPA symbol in each line
    of the source file when the line is split on \t
    * xsampa_ind (int): Same, but the index of the XSAMPA symbol
    '''
    xs2ipa = {}
    with open(i2x_path) as m_f:
        for line in m_f:
            phone_symbols = line.split('\t')
            ipa_symbol, xsampa_symbol = phone_symbols[ipa_ind], phone_symbols[xsampa_ind]
            xs2ipa[xsampa_symbol.strip()] = ipa_symbol.strip()
    return xs2ipa


def x2i_prondict(src_path: str, out_path: str, xs2ipa: dict = None):
    '''
    Convert a pronounciation dictionary from XSAMPA format
    to IPA format.

    Input arguments:
    * src_path (str): The path of the XSAMPA pronounciation dictioanary
    where each line consists of "<word>\t<XSAMPA pron>"
    * out_path (str): The target path for the IPA pronounciation dictionary
    where each line follows the same format.
    * xs2ipa (dict or None): A mapping from XSAMPA to IPA
    '''
    if xs2ipa is None:
        xs2ipa = x2i_map()
    with open(src_path) as i_f, open(out_path, 'w') as o_f:
        for line in i_f:
            word, pron = line.split('\t')
            o_f.write("{}\t{}\n".format(word, xs2ipa_str(pron, xs2ipa)))


def xs2ipa_str(xs_string: str, xs2ipa: dict):
    '''
    Map a space seperated string of XSAMPA symbols to a corresponding
    string of space seperated IPA symbols

    Input arguments:
    * xs_string (str): The XSAMPA symbols to be converted
    * xs2ipa (dict): A mapping from XSAMPA to IPA symbols
    '''
    return " ".join(xs2ipa[p] for p in xs_string.strip().split())
