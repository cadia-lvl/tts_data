import time
start = time.time()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from xml.etree import ElementTree as  ET
import re
import os
import pickle

NS = {'a': 'http://www.tei-c.org/ns/1.0'}
def to_sentence(s_node, punctuation=None, lower_case=None, ns=NS):
    """Convert w and c token nodes of a s node to a ws seperated string of
    tokens"""
    tokens = []
    for node in s_node.findall('./*'):
        tokens.append((node.text, node.attrib['type']))
    charlist = []
    tokens_len = len(tokens)
    for i, token in enumerate(tokens):
        if lower_case == 'all' or (lower_case == 'not_ne'
                                   and not (token[1][0] == 'n'
                                            and token[1][-1] == 's')):
            token = (token[0].lower(), token[1])
        if token[1] == 'punctuation' and punctuation == 'remove':
            token = ('', token[1])
        charlist.extend(token[0])
        if (i != tokens_len - 1 and (tokens[i+1][1] != 'punctuation'
                                     or punctuation not in ['join'])
            and not (token[1] == 'punctuation' and punctuation == 'remove')):
            charlist.extend(' ')
        sentences = ''.join(charlist)
    # No sentences not ending in punctuation
    if not token[1] == 'punctuation' or token[1] == 'x' or token[1] == 'as':
        sentences = ""
    return sentences

def parse(punct, lc, fobj):
    try:
        docList = []
        """Parse TEI XML file, and print out sentences"""
        fmt = '{sent}'
        root = ET.parse(fobj).getroot()
        fid = root.find('.//a:fileDesc//a:idno', NS).text
        for p in root.findall('.//a:body//a:p', NS):
            pid = p.attrib['n']
            for s in p.findall('.//a:s', NS):
                sid = s.attrib['n']
                sent = to_sentence(s, punctuation=punct, lower_case=lc)
                tmp = fmt.format(pid=pid, sid=sid, fid=fid, sent=sent)
                docList.append(tmp)
    except:
        pass
    return docList

def regexList(parsed):
    """We don't want foreign letters or anything that would need to be normalized (such as numbers and abbreviations)"""
    reg = "[^aábdðeéfghiíjklmnoóprstuúvxyýþæöAÁBDÐEÉFGHIÍJKLMNOÓPRSTUÚVXYÝÞÆÖ\s,/()„“‘:;&?!!\"|(.“)]"
    regList = [item for item in parsed if re.search(reg, item) == None and item != ""]
    return regList

def list_files(directory):
    fileAll = []
    for root, dirs, files in os.walk(directory):
        fileList = [os.path.join(root, name) for name in files if name.endswith(".xml")]
        fileAll.extend(fileList)
    return fileAll

def main():
    folders = ['althingislog', 'bylgjan', 'domstolar', 'haestirettur', 'ras1', 'ras1_og_2', 'ras2',
              'silfuregils', 'sjonvarpid', 'stod2', 'visir']
    #folders = ['bylgjan']
    allSentencesFolder = []
    for folder in folders:
        xmlSources = list_files('rmh2/CC_BY/' + folder)
        allSentences = []
        for filename in xmlSources:
            regList = regexList(parse('join', 'none', open(filename)))
            allSentences.extend(regList)
        allSentencesFolder.extend(allSentences)
        with open ((f'rmh_{folder}.pickle'), 'wb') as fp:
            pickle.dump(allSentencesFolder, fp)
            print(len(allSentencesFolder))
            print(time.time() - start)
    allSentencesUnique = list(set(allSentencesFolder))
    print(len(allSentencesUnique))
    with open('rmh_demo2_allSources.pickle', 'wb') as fp:
        pickle.dump(allSentences, fp)

    return allSentencesUnique

asu = main()
end = time.time() - start

print(end)