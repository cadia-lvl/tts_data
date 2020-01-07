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


import os
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from misc.xml_tools import parse
from tqdm import tqdm

from bin_tools import BinVerifer
from conf import ICE_ALPHABET, OTHER_CHARS


def gs_to_files(src_dir: str, out_dir: str):
    '''
    Given the directory of the GullStadall dataset, this
    function will convert tokens to the standard format
    used in this project. The output will be stored under
    out_dir. Data from each subdirectory from Gullstadall
    will be stored together in a .txt file with the same
    name, e.g. blog/ -> blog.txt

    Input arguments:
    * src_dir (str): The main directory containing
    Gullstadall
    * out_dir (str): The target directory to store the
    formatted data
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname in tqdm(os.listdir(src_dir)):
        if os.path.isfile(os.path.join(src_dir, fname)):
            with open(os.path.join(src_dir, fname), 'r') as i_f,\
                    open(os.path.join(out_dir, fname), 'w') as o_f:
                token, idx = '', 0
                for line in i_f:
                    if not line.strip():
                        # new sentence
                        o_f.write('{}\n'.format(token))
                        token, idx = '', 0
                    else:
                        w, _ = line.split('\t')
                        if w in OTHER_CHARS or idx == 0:
                            token += w
                        else:
                            token += ' {}'.format(w)
                        idx += 1


def hlbs_to_files(src_dir: str, out_dir: str):
    '''
    Given the directory of the Hljodbokasafn dataset, this
    function will convert tokens to the standard format
    used in this project. The output will be stored under
    out_dir. Data from each subdirectory from Hljodbokasafn
    will be stored together in a .txt file with the same
    name, e.g. hlbsf_101_int/ -> hlbsf_101_int.txt

    Input arguments:
    * src_dir (str): The main directory containing
    the Hljodbokasafn dataset
    * out_dir (str): The target directory to store the
    formatted data
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for directory in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir, directory)):
            with open(
                os.path.join(out_dir, '{}.txt'.format(directory)), 'w')
            as o_f:
                for fname in os.listdir(
                        os.path.join(src_dir, directory, 'text')):
                    if os.path.isfile(os.path.join(
                            src_dir, directory, 'text', fname)):
                        o_f.write(open(os.path.join(
                            src_dir,
                            directory, 'text', fname)).readline())


def ivona_to_file(src_dir: str, out_path: str):
    '''
    Given the directory of the Ivona dataset, this
    function will convert tokens to the standard format
    used in this project. The output will be stored in
    a single .txt file at out_path

    Input arguments:
    * src_dir (str): The main directory containing
    the Ivona dataset
    * out_path (str): The target path to store the
    formatted data
    '''
    with open(out_path, 'w') as o_f:
        for fname in tqdm(os.listdir(src_dir)):
            o_f.write(open(os.path.join(
                src_dir, fname)).readline())


def malromur_to_file(
        db_path: str, out_path: str, table_name='malromur_II',
        column='corrected_prompt'):
    '''
    Given a path the sqlite dump of the Malromur dataset,
    this function will convert tokens to the standard format
    used in this project. The output will be stored in
    a single .txt file at out_path

    Input arguments:
    * db_path (str): A path to a sqlite dump of malromur
    * out_path (str): The target path to store the
    formatted data
    * table_name (str): The name of the data table in the
    sqlite dump
    * column (str): The column from <table_name> that contains
    the token.
    '''
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sqlite_select_query = """SELECT {} FROM {}""".format(
            column, table_name)
        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()

        with open(out_path, 'w') as out_file:
            for row in tqdm(records):
                out_file.write('{}\n'.format(row[0]))
        cursor.close()

    except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
    finally:
        if conn:
            conn.close()
            print("The SQLite connection is closed")


def rmh_to_file(src_path: str, out_path: str):
    '''
    Input arguments:
    * src_path (str): The file containing
    the Risamalheild dataset excerpt
    * out_path (str): The target path to store the
    formatted data
    '''
    with open(src_path) as i_f, open(out_path, 'w') as o_f:
        for line in tqdm(i_f):
            txt_id, txt = line.split('\t')
            o_f.write(f'{txt.strip()}\trmh-{txt_id}\n')


def rmh_parser(src_dir: str, out_dir: str):
    '''
    All .xml files under src_dir/src/<year>/<month> will
    be written to out_dir/src-<year>-<month>.txt

    No specific preprocessing is done here, except that sentences
    that contain '.' punctuation anywhere except for at the end
    of a sentence, are removed.

    Input arguments:
    * src_dir (str): The directory of the RMH data
    * out_dir (str): The target directory to write out
    the new RMH parsed data.
    '''
    sentences = defaultdict(list)

    executor = ProcessPoolExecutor()
    futures = []

    f_count = 0
    for _ in os.walk(src_dir):
        f_count += 1

    for dirName, subdirList, fileList in tqdm(os.walk(src_dir), total=f_count):
        for fname in fileList:
            try:
                source = Path(dirName).parent.parent.name
                sentences[source] += parse(os.path.join(dirName, fname))
                futures.append([
                    source,
                    executor.submit(
                        partial(parse, os.path.join(dirName, fname)))])
            except Exception as e:
                print(e, os.path.join(dirName, fname))
                continue
    for (src, text) in [
            (future[0], future[1].result()) for future in tqdm(futures)]:
        sentences[src] += text
    for source, sents in sentences.items():
        with open(f'{os.path.join(out_dir, source)}.txt', 'w') as o_f:
            for sent in sents:
                o_f.write(f'{sent}\t{source}\n')


def tokens_to_file(src_dir: str, out_path: str):
    '''
    Given a directory containing possibly many .txt files
    as well as subdirectories containing more .txt files of
    tokens in the standard format, create a single token file
    from all found .txt files.

    Input arguments:
    src_dir (str): The directory containing all dataset sources
    in the standard format
    out_path (str): The target path where the unified output
    will be stored
    '''
    with open(out_path, 'w') as o_f:
        for path in tqdm(os.listdir(src_dir)):
            tok_path = os.path.join(src_dir, path)
            if os.path.isdir(tok_path):
                # Handle directories
                for subpath in os.listdir(tok_path):
                    file_path = os.path.join(tok_path, subpath)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as i_f:
                            for line in i_f:
                                o_f.write('{}\t{}-{}\n'.format(
                                    line.strip(), path, Path(subpath).stem))
            else:
                # Handle single files
                with open(tok_path, 'r') as i_f:
                    for line in i_f:
                        o_f.write('{}\t{}\n'.format(
                            line.strip(), Path(path).stem))


def preprocess(
        src_path: str, out_path: str, bad_path: str,
        bin_ver=BinVerifer(), min_char_length: int = 10,
        max_char_length: int = None, min_word_length: int = 5,
        max_word_length: int = 15, contains_pron: bool = False):
    '''
    Given a path to a file where each line is either
    <sentence>\t<source_id>\t<transcription>
    or
    <sentence>\t<source_id>
    this function will perform a preprocessing step and output
    the same lines into two files at out_path if they pass
    the preprocessing step or bad_path if not.

    Input arguments:
    * src_path (str): The path to the source file containing
    lines to be preprocessed
    * out_path (str): The path to the file where lines that
    pass the preprocessing step should be stored
    * bad_path (str): The path to the file where lines that
    do not pass the preprocessing step should be stored
    * bin_ver (bin_tools.BinVerifier = BinVerifier()): An
    instance of the BinVerifier class.
    * min_char_length (int or None = 10): The minimum number
    of characters a sentence needs to contain to pass.
    * max_char_length (int or None = None): The maximum
    number of characters a sentence can contain to pass.
    * min_word_length (int or None = 5): The minimum number
    of words a sentence needs to contain to pass.
    * max_word_length (int or None = 15): The maximum number
    of words a sentence can contain to pass.
    * contains_pron (bool = False): Is True if each line
    contains a transcription
    '''
    deny_pattern = re.compile(r'[^{}{}{}]'.format(
        ICE_ALPHABET, ICE_ALPHABET.upper(), OTHER_CHARS)).search
    num_lines = len(open(src_path).readlines())
    with open(src_path, 'r') as i_f, open(out_path, 'w') as o_f,\
            open(bad_path, 'w') as b_f:
        for line in tqdm(i_f, total=num_lines):
            if contains_pron:
                token, src, *pron = line.split('\t')
            else:
                token, src = line.split('\t')
                src = src.strip()
            # check if too short
            if (min_char_length and len(token) < min_char_length)
            or (min_word_length and len(token.split()) < min_word_length):
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'SHORT'))
            # check if too long
            elif (max_char_length and len(token) > max_char_length)
            or (max_word_length and len(token.split()) > max_word_length):
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'LONG'))
            # check if starts with a capital letter
            elif token[0] not in ICE_ALPHABET.upper():
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'CAPITAL'))
            # check if any character not in deny_pattern
            elif bool(deny_pattern(token)):
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'ILLEGAL'))
            # check if "." is anywhere in sentence except at the end
            elif token.find('.') not in [-1, len(token) - 1]:
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'PUNC'))
            # check if it contains either „ or “ without the other
            elif ('„' in token and '“' not in token)
            or ('„' not in token and '“' in token):
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'GOOSE'))
            # lastly check if in BIN
            elif not bin_ver.check_utt(token):
                b_f.write('{}\t{}\t{}\n'.format(token, src, 'BIN'))
            else:
                o_f.write(line)
