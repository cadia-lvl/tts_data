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


def x2i_diphones(i2x_path='./data/ipa_2_xsampa.txt', ipa_ind:int=0,
    xsampa_ind:int=2, xdp_path:str='./data/diphones/diphones_xsampa.txt',
    out_path:str='./data/diphones/diphones_ipa.txt'):
    '''
    Convert a list of XSAMPA diphones to a list of IPA diphones. Additionally
    if the symbol '_' appears in the diphone we skip the diphone.

    Input arguments:
    * i2x_path (str): The path to the IPA XSAMPA index
    * ipa_ind (int): The index of the IPA symbol in each line
    of the source file when the line is split on \t
    * xsampa_ind (int): Same, but the index of the XSAMPA symbol
    xdp_path (str): The path to the list of XSAMPA diphones where
    each line is a tab seperated tuple.
    * out_path (str): The target path to store the IPA output in
    the same format
    '''
    xs2ipa = {}
    with open(i2x_path) as m_f:
        for line in m_f:
            ps = line.split('\t')
            i, x = ps[ipa_ind], ps[xsampa_ind]
            xs2ipa[x.strip()] = i.strip()

    result = []
    with open(xdp_path) as i_f, open(out_path, 'w') as o_f:
        for line in i_f:
            p1, p2 = line.split(' ')
            if '_' not in (p1.strip(), p2.strip()):
                o_f.write("{}\t{}\n".format(
                    xs2ipa[p1.strip()], xs2ipa[p2.strip()]))

def x2i_map(src_path:str='./data/ipa_2_xsampa.txt', ipa_ind:int=0,
    xsampa_ind:int=1) -> dict:
    '''
    Generate a mapping from XSAMPA to IPA in the
    form of a dictionary. This function expects a
    tab seperated file of IPA, XSAMPA symbols per line.

    Input arguments:
    * src_path (str): The path to the IPA XSAMPA index
    * ipa_ind (int): The index of the IPA symbol in each line
    of the source file when the line is split on \t
    * xsampa_ind (int): Same, but the index of the XSAMPA symbol
    '''
    xs2ipa = {}
    with open(src_path) as m_f:
        for line in m_f:
            ps = line.split('\t')
            i, x = ps[ipa_ind], ps[xsampa_ind]
            xs2ipa[x.strip()] = i.strip()
    return xs2ipa

def x2i_prondict(src_path:str, out_path:str, xs2ipa=x2i_map()):
    '''
    Convert a pronounciation dictionary from XSAMPA format
    to IPA format.

    Input arguments:
    * src_path (str): The path of the XSAMPA pronounciation dictioanary
    where each line consists of "<word>\t<XSAMPA pron>"
    * out_path (str): The target path for the IPA pronounciation dictionary
    where each line follows the same format.
    * xs2ipa (dict): A mapping from XSAMPA to IPA
    '''
    with open(src_path) as i_f, open(out_path, 'w') as o_f:
        for line in i_f:
            w, pron = line.split('\t')
            o_f.write("{}\t{}\n".format(w, xs2ipa_str(pron, xs2ipa)))

def xs2ipa_str(xs_string:str, xs2ipa:dict):
    '''
    Map a space seperated string of XSAMPA symbols to a corresponding
    string of space seperated IPA symbols

    Input arguments:
    * xs_string (str): The XSAMPA symbols to be converted
    * xs2ipa (dict): A mapping from XSAMPA to IPA symbols
    '''
    return " ".join(xs2ipa[p] for p in xs_string.strip().split())