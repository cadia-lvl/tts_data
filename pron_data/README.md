## all_tokens_norm_clean_g2p.txt (80417 tokens)
A tab seperated list where each line is like `<token> \t <source> \t <phones>` where the source is a rudementary identifier for the source of the token and phones is a tab-seperated list of word-phone-predictions, each of which is space seperated. An example:

 Token | Source | Phones
-------|--------|--------
 Kettir verði ekki bundnir í árborg | malromur | cʰ ɛ h t ɪ r \<\t\> v ɛ r ð ɪ \<\t\> ɛ h c ɪ \<\t\> p ʏ n t n ɪ r \<\t\> iː \<\t\> au r p ɔ r k
 Níu grunaðir um ölvunarakstur |malromur | n i j ʏ \<\t\> k r ʏː n a ð ɪ r \<\t\> ʏ m \<\t\> œ l v ʏ n a r a k s t ʏ r

## Diphone coverage comparisons
Using the complete list of valid diphones in `data/diphones/complete_ipa.txt`, the tokens in all_tokens_norm_clean_g2p.txt cover approx 81% of the diphones. The ivona dataset alone (8909 tokens), approx 10x smaller than the complete list, covers approx 76 % of the listed valid diphones.