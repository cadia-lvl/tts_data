import os
import re
import string
from tqdm import tqdm
import sqlite3
from pathlib import Path

from sequitur_tools import get_phones

NUMS = '0123456789'
CHARS = string.ascii_letters
ICE_CHARS = 'áéíóúýæöþð'
ICE_CHARS += ICE_CHARS.upper()
ONLY_ENGLISH = 'cqw'
ONLY_ENGLISH += ONLY_ENGLISH.upper()

start_pattern = re.compile('[^a-zA-Z0-9{}]'.format(ICE_CHARS))
toksub_pattern = re.compile(r'[^a-zA-Z0-9{}\s,/()„“‘:;&?!\"(.“)]'.format(ICE_CHARS))

def gs_to_files(src_dir:str, out_dir:str):
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
                        if w in ['.', ',', '?', '!', ':'] or idx == 0:
                            token += w
                        else:
                            token += ' {}'.format(w)
                        idx += 1

def hlbs_to_files(src_dir:str, out_dir:str):
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
            with open(os.path.join(out_dir,
                '{}.txt'.format(directory)), 'w') as o_f:
                for fname in os.listdir(
                    os.path.join(src_dir, directory, 'text')):
                    if os.path.isfile(os.path.join(
                        src_dir, directory, 'text', fname)):
                        o_f.write(open(os.path.join(src_dir,
                            directory, 'text', fname)).readline())

def ivona_to_file(src_dir:str, out_path:str):
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

def malromur_to_file(db_path:str, out_path:str,
    table_name='malromur_II', column='corrected_prompt'):
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


def tokens_to_file(src_dir:str, out_path:str):
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
                        o_f.write('{}\t{}\n'.format(line.strip(), Path(path).stem))


def common_normalize(src_path:str, out_path:str):
    '''
    Apply the normalize function to each token in a
    list of tokens

    Input arguments:
    * src_path (str): The path to the list of tokens
    * out_path (str): The target path where the normalized
    output will be stored.
    '''
    with open(src_path, 'r') as i_f, open(out_path, 'w') as o_f:
        for line in i_f:
            token, src = line.split('\t')
            o_f.write('{}\t{}'.format(normalize(token), src))

def normalize(token:str, lower_all:bool=True):
    '''
    Normalize a text token. This function will:
    * Remove any symbols not found in the global
    toksub_pattern regex pattern
    * Remove any leading symbols that are neither
    characters nor numericals.
    * Lowercase any uppercase characters that do not
    appear at the start of words.

    Input arguments:
    * token (str): The token to normalize
    * lower_all (bool=True): If True, all characters are made
    lowercase, otherwise characters at the start of words are
    allowed to be uppercase.
    '''
    token = re.sub(toksub_pattern, '', token)
    while len(token) != 0 and token[0] not in CHARS+ICE_CHARS+NUMS:
        token = token[1:]
    if lower_all:
        token = token.lower()
    else:
        # remove bad uppercase
        l_token = list(token)
        for i in range(1, len(l_token)):
            if l_token[i].isupper() and l_token[i] != ' ':
                l_token[i] = l_token[i].lower()
        token = ''.join(l_token)
    token = token.strip()
    return token


def split_bad_tokens(src_path:str, out_path:str, bad_path:str,
    min_char_length=10, max_char_length=None, min_word_length=3,
    max_word_length=None):
    '''
    Given a list of tokens, remove all tokens that check
    any of these boxes:
        * contains numericals
        * shorter than 10 characters
        * shorter than 3 words
        * with "." somewhere else than at the end of the token
        * containing non-icelandic characters from the
        english alphabet

    Input arguments:
    * src_path (str): A path to the file containing a list of
    tokens
    * out_path (str): A target path where the filtered output
    will be stored
    * bad_path (str): A target path where tokens that check any
    of the above boxes will be stored.
    * min_char_length (None/int): The minimum length of tokens
    * max_char_length (None/int): The maximum length of tokens
    '''
    with open(src_path, 'r') as i_f, open(out_path, 'w') as o_f,\
        open(bad_path, 'w') as b_f:
        for line in i_f:
            token, src = line.split('\t')
            if any(c.isdigit() for c in token):
                b_f.write('{}\t{}\t{}\n'.format(token, src.strip(), 'DIGITS'))
            elif (min_char_length and len(token) < min_char_length) or \
                (min_word_length and len(token) < min_word_length):
                b_f.write('{}\t{}\t{}\n'.format(token, src.strip(), 'SHORT'))
            elif (max_char_length and len(token) > max_char_length) or \
                (max_word_length and len(token) < max_word_length):
                b_f.write('{}\t{}\t{}\n'.format(token, src.strip(), 'LONG'))
            elif token.find('.') not in [-1, len(token) - 1]:
                b_f.write('{}\t{}\t{}\n'.format(token, src.strip(), 'PUNC'))
            elif any(c in ONLY_ENGLISH for c in token):
                b_f.write('{}\t{}\t{}\n'.format(token, src.strip(), 'ENGLISH'))
            else:
                o_f.write('{}\t{}\n'.format(token, src.strip()))