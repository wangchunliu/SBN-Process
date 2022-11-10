#!/bin/bash
# Preprocessing scripts for PMB SBN data

### Get data (raw-sbn and corresponding sentences) from PMB website
# In the root directory
wget "https://pmb.let.rug.nl/releases/pmb-4.0.0.zip"
unzip pmb-4.0.0.zip
cd pmb-4.0.0/data

### Merge SBN files to one file: gold, silver and bronze
for lang in en de it nl; do
    for type in gold silver bronze; do
        cd ${lang}/${type}
        find ./${*}/${*}/ -type f -name "en.drs.sbn" | xargs -I{} sh -c "cat {}; echo ''" > sbn.txt
        python ~/SBN-Process/0_get_raw_file.py -i sbn.txt -lang ${lang} -ipath ./ -o train.txt
        cd -
    done
done


### Split gold data to train, dev and test


### Process clause-SBN to linearized-SBN (for seq2seq models)


### Process linearized-SBN to Graph-structure SBN (for Graph2seq models)

