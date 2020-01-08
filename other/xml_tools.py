"""XML parser for RMH"""
# -*- coding: utf-8 -*-

# Copyright 2015, 2018 Róbert Kjaran <robert@kjaran.com>
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


from xml.etree import ElementTree as ET
import sys
import argparse

from conf import MAX_WORDS, MIN_WORDS


def to_sentence(s_node):
    """Convert w and c token nodes of a s node to a ws seperated string of
    tokens"""
    tokens = []
    for node in s_node.findall('./*'):
        tokens.append((node.text, node.attrib['type']))

    charlist = []
    tokens_len = len(tokens)
    for i, token in enumerate(tokens):
        charlist.extend(token[0])
        if (i != tokens_len - 1 and (tokens[i+1][1] != 'punctuation')):
            charlist.extend(' ')

    return ''.join(charlist)


def parse(fobj):
    """Parse TEI XML file, and print out sentences"""
    root = ET.parse(fobj).getroot()
    sentences = []
    for paragraph in root.findall(
            './/a:body//a:p', {'a': 'http://www.tei-c.org/ns/1.0'}):
        for sentence in paragraph.findall('.//a:s', {'a': 'http://www.tei-c.org/ns/1.0'}):
            sentence = to_sentence(sentence)

            sen_len = len(sentence.split())
            if MIN_WORDS <= sen_len <= MAX_WORDS:
                sentences.append(sentence)
    return sentences


def main():
    """Argument parser for the XML parser"""
    parser = argparse.ArgumentParser(
        description="""Parse sentences from Icelandic Tagged Corpus
        (MÍM) TEI XML files""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--print-ids', action='store_true',
        help='First column is the sentence id (filename-pid-sid)')
    parser.add_argument(
        '--punctuation', choices=['noop', 'remove', 'join'],
        default='noop', help='What to do with punctuation.')
    parser.add_argument(
        '--lower-case', choices=['none', 'all', 'not_ne'],
        default='none',
        help="""Do case modification. Specifying "none" does
        nothing, "all" makes all text lowercase and "not_ne"
        skips all words marked as named entities (so it doesn't
        help with multi-word NEs). """)
    parser.add_argument(
        'input', type=argparse.FileType('r'), nargs='+',
        help="""Input files can be either TEI XML files or text
        files with one path to a TEI XML file per line.""")
    parser.add_argument(
        '--output', type=argparse.FileType('w'),
        default=sys.stdout, help='Path to output file.')

    args = parser.parse_args()

    for ins in args.input:
        try:
            parse(ins)
        except ET.ParseError:
            ins.seek(0)
            for line in ins:
                with open(line.strip(), 'r') as fobj:
                    parse(fobj)
