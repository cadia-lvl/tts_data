## dipones_xsampa.txt
Contains a tab-seperated list of valid diphones in a rather proprietary XSAMPA format sourced from this Master thesis : https://notendur.hi.is/eirikur/mt/Towards%20Speech%20Synthesis%20for%20Icelandic.pdf . This list was based on the ones used to record the MBROLA synthesiser and is extensive, but not complete.

## diphones_ipa.txt
Contains a tab-seperated list of the same diphones as in diphones_xsampa.txt but in IPA format

## denied.txt
Contains a tab-seperated list of IPA diphones (only inter-word diphones) that a Sequitur G2P model predicted and did not appear on the diphones_ipa.txt list. This list contains 272 possibly valid diphones.

## good_ipa.txt & bad_ipa.txt
good_ipa.txt contains a tab-seperated list of IPA diphone sourced from denied.txt and verified by diphone lookup in the Prondict. bad_ipa.txt is the rest from denied.txt and might contain some valid diphones.

## complete_ipa.txt
Contains a concatenation of diphones_ipa.txt and good_ipa.txt. This list contains 2101 diphones.

