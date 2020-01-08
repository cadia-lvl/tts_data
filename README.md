# Tools for generating TTS datasets with good diphone coverage for the Icelandic language.
This repository is mixed with both multiple G2P models as well as text parsers / normalize steps / phonetic stuff and more. Here is a brief introduction of the more notable stuff.

## `sequitur_tools.py`
This file contains both class-based and function-based python-sequitur buindings and more.

##  `x2i_tools.py`
This file contains multiple methods for easily mapping between XSAMPA and IPA and is highly coupled to the work that was performed.

## `utt_tools.py`
This file contains methods for parsing and normalizing text utterances from the following datasets:
* ivona
* Gullstadall
* The Hljodbokasafn TTS dataset
* Malromur

## `score.py`
This file contains method for ordering a collection of utterances by a heuristic that should increase diphone coverage. This is still a work in progress.

## `prondata.py`
This file contains many methods that can be used for analyzing diphones in the set, coverage, missing diphones and more.

## `torchG2P/` : A Sequence-2-sequence neural G2P model in PyTorch
### Setup
Install dependencies via `pip install -r requirements.txt`.

### Easily run the G2P model
* Make sure dependencies are installed
* Create a directory `./data` and place the pronounciation dictionary there, e.g. `./data/prondict_ice.txt`. An Icelandic pronounciation dictionary is available (as of writing this) at `terra:/data/prondict_sr/v1/frambordabok_asr_v1.txt `
* Run `python3 main.py` which will use default arguments. Look at the documented code to make changes to parameters.
* Results will be default be placed under `./results/<experiment_name>` in the form of a torch state dictionary, `mdl.ckpt`. Currently logging is only in the form of printing using e.g. `print(...)`. To store logs run `python3 main.py > log.txt` to save the logs.

# Licence
Copyright (c) Atli Thor Sigurgeirsson <atlithors@ru.is>. All rights reserved

Licensed under the [Apache 2.0](LICENSE) license.