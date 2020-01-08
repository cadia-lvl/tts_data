# Module for preparing text data for TTS data collections, specifically for the Icelandic language.

## Installation
The code should be mostly supported by any Python 3.* but Python 3.6 or higher is recommended. Install dependencies by simply running `pip install -r requirements.txt`

## Introduction
The code includes methods helpful for all major steps in the text collection side of TTS data collecting:
1. Gathering data: [`utt_tools.py`](utt_tools.py) contains methods for parsing utterances from some of the open Icelandic text datasets.
2. Preprocessing: A preprprocessing routine is also found in [`utt_tools.py`](utt_tools.py), designed specifically for these datasets.
3. Grapheme-to-Phoneme prediction: A Sequitur wrapper is found in [`sequitur_tools.py`](sequitur_tools.py). A pretrained model file is also included in this repository under [`pron_data`](pron_data). The model path is configured in [`conf.py`](conf.py).
4. Scoring: [`prondata.py`](prondata.py) contains a large class, `PronData.py`. It has multiple methods that can prove convenient to others but most important is `Prondata.score()` which scores the utterances based on length and diphone density.


## Testing
Simple tests are found in [`tests.py`](tests/tests.py) which additionally demonstrate some of the main operations in this module.

## Reading lists
Under [`reading_lists`](reading_lists) are 4 reading lists, varying in length. [`rl_full.txt`](reading_lists/rl_full.txt) contains a very high diphone coverage, containing almost 20 instances of every valid Icelandic diphone.

# Licence
Copyright (c) Atli Thor Sigurgeirsson <atlithors@ru.is>. All rights reserved

Licensed under the [Apache 2.0](LICENSE) license.