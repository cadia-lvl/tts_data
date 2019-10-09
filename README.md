# Setup
Install dependencies via `pip install -r requirements.txt`.  

# Easily run the G2P model
* Make sure dependencies are installed
* Create a directory `./data` and place the pronounciation dictionary there, e.g. `./data/prondict_ice.txt`. An Icelandic pronounciation dictionary is available (as of writing this) at `terra:/data/prondict_sr/frambordabok_asr_v1.txt `
* Run `python3 main.py` which will use default arguments. Look at the documented code to make changes to parameters.
* Results will be default be placed under `./results/<experiment_name>` in the form of a torch state dictionary, `mdl.ckpt`. Currently logging is only in the form of printing using e.g. `print(...)`. To store logs run `python3 main.py > log.txt` to save the logs.