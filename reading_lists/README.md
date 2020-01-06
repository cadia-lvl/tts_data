# Information about reading lists
The complete list is found in `full.txt` and contains 24399 sentences. 3 other lists are published that vary in estimated reading length.

| name | number of sentences |
| --- | --- |
| small | 720 |
| medium | 7200 |
| long | 14400 |

Each line in each file has the same format and contains 5 or more tab seperated items. Here is an example from `small_1hour.txt`:

| sentence | source | score | transcription |
| --- | --- | --- | --- | --- |
| Þetta hefði allt átt sér stað í geðæsingu. | domstolar | 0.02454774620758596 | ~ θ ɛ h t a	h ɛ v ð ɪ	a l̥ t	au h t	s j ɛː r	s t a ð	iː	c ɛ aiː s i ŋ k ʏ ~

Here is an example python snippet, showing how one could read from these files:
```
with open('small_1hour.txt') as f:
        for line in f:
            sentence, source, score, *transcription = line.split('\t')
```
Here, `transcription` is read in as a list of transcriptions for each word in the sentence.