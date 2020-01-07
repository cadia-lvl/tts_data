# -*- coding: utf-8 -*-
#
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


from conf import VALID_DIPHONES_PATH

phonemes = [
    'a', 'r', 't', 's', 'n', 'ɪ', 'l', 'ʏ', 'k', 'm',
    'ð', 'ɛ', 'v', 'p', 'h', 'f', 'j', 'c', 'i', 'ɔ', 'r̥',
    'ei', 'ŋ', 'ɣ', 'ou', 'œ', 'ouː', 'au', 'ai', 'aː', 'auː',
    'iː', 'eiː', 'ɪː', 'ɛː', 'θ', 'l̥', 'tʰ', 'uː', 'aiː',
    'kʰ', 'u', 'ɔː', 'x', 'œː', 'œy', 'n̥', 'cʰ', 'œyː', 'pʰ',
    'ɲ', 'ʏː', 'ç', 'ŋ̊', 'm̥', 'ʏi', 'ɲ̊', 'ɔi']

sil_phone = '~'
sil_phonemes = phonemes
sil_phonemes.append(sil_phone)
sil_diphones = []
diphones = []
unique_diphones = set()
list_diphones = []

with open(VALID_DIPHONES_PATH) as i_f:
    for line in i_f:
        p1, p2 = line.split('\t')
        diphones.append("".join([p1, p2.strip()]))

        sil_diphones.append("".join([p1, p2.strip()]))

        list_diphones.append((p1, p2))

        unique_diphones.add(p1)
        unique_diphones.add(p2)

for dp in unique_diphones:
    sil_diphones.append("".join([dp.strip(), sil_phone]))
    sil_diphones.append("".join([sil_phone, dp.strip()]))
