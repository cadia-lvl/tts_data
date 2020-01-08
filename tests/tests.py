"""Tests for the more important routines in this work"""
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

import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conf import ROOT_DIR
from sequitur_tools import g2p_file
from prondata import PronData


TEST_DIR = ROOT_DIR / 'tests'
OUTPUT_DIR = TEST_DIR / 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class G2PTest(unittest.TestCase):
    """
    Test the Sequitur G2P model agains a precomputed list
    """
    def test(self):
        """Predict transcription and compare to precomputed transcription"""
        print(f"Testing G2P, see {OUTPUT_DIR} for results")
        g2p_file(
            TEST_DIR / 'unscored_list_without_pron.txt',
            OUTPUT_DIR / 'unscored_list_with_pron.txt')

        with open(OUTPUT_DIR / 'unscored_list_with_pron.txt', 'r') as generated,\
                open(TEST_DIR / 'unscored_list.txt', 'r') as truth:
            generated_lines, truth_lines = generated.readlines(), truth.readlines()
            for i, line in enumerate(generated_lines):
                self.assertEqual(line, truth_lines[i])

class ScoreTest(unittest.TestCase):
    """
    Test the greedy scoring method in prondata.PronData
    """
    def test(self):
        """Compute scores and compare to real scores"""
        print(f"Testing greedy score, see {OUTPUT_DIR} for results")
        dataset = PronData(TEST_DIR / 'unscored_list.txt')
        dataset.score(OUTPUT_DIR / 'scored_list.txt')

        with open(OUTPUT_DIR / 'scored_list.txt', 'r') as generated,\
                open(TEST_DIR / 'scored_list.txt', 'r') as truth:
            generated_lines, truth_lines = generated.readlines(), truth.readlines()
            for i, line in enumerate(generated_lines):
                self.assertEqual(line, truth_lines[i])

if __name__ == '__main__':
    unittest.main()
