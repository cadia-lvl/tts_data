# -*- coding: utf-8 -*-


from conf import VALID_DIPHONES_PATH
VALID_DIPHONES_PATH = './pron_data/diphones/complete_ipa.txt'

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
list_diphones  = []
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