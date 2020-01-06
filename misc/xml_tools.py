#!/usr/bin/env python3
#
# Copyright 2015, 2018 Róbert Kjaran <robert@kjaran.com>
#

from xml.etree import ElementTree as  ET
import sys

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
        charlist.extend(token[0])
        if (i != tokens_len - 1 and (tokens[i+1][1] != 'punctuation')):
            charlist.extend(' ')

    return ''.join(charlist)


def parse(fobj, punctuation=None, lower_case=None):
    """Parse TEI XML file, and print out sentences"""
    root = ET.parse(fobj).getroot()
    sentences = []
    for p in root.findall('.//a:body//a:p', NS):
        for s in p.findall('.//a:s', NS):
            sent = to_sentence(s, punctuation=punctuation,
                lower_case=lower_case)

            senlen = len(sent.split())
            if 5 <= senlen and senlen <= 15:
                sentences.append(sent)
    return sentences

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Parse sentences from Icelandic Tagged Corpus (MÍM) TEI XML files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--print-ids', action='store_true',
                        help='First column is the sentence id (filename-pid-sid)')
    parser.add_argument('--punctuation', choices=['noop', 'remove', 'join'],
                        default='noop', help='What to do with punctuation.')
    parser.add_argument('--lower-case', choices=['none', 'all',
                                                 'not_ne'],
                        default='none',
                        help="""Do case modification. Specifying "none" does
                        nothing, "all" makes all text lowercase and "not_ne"
                        skips all words marked as named entities (so it doesn't
                        help with multi-word NEs). """)
    parser.add_argument('input', type=argparse.FileType('r'), nargs='+',
                        help="""Input files can be either TEI XML files or text
                        files with one path to a TEI XML file per line.""")
    parser.add_argument('--output', type=argparse.FileType('w'),
                        default=sys.stdout, help='Path to output file.')

    args = parser.parse_args()

    for ins in args.input:
        try:
            parse(ins, ags.punctuation, args.lower_case)
        except ET.ParseError:
            ins.seek(0)
            for line in ins:
                with open(line.strip(), 'r') as fobj:
                    parse(args, fobj)


if __name__ == '__main__':
    main()