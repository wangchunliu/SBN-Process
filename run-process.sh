#!/bin/bash
# Preprocessing scripts for PMB SBN data
mkdir -p ~/SBN-data
cd ~/SBN-data 

### 1.Get data (raw-sbn and corresponding sentences) from PMB website
wget "https://pmb.let.rug.nl/releases/pmb-4.0.0.zip"
unzip pmb-4.0.0.zip
cd pmb-4.0.0/data


### 2.Merge SBN files to one file: gold, silver and bronze
for lang in en de it nl; do
    for type in gold silver bronze; do
        cd ${lang}/${type}
        find ./${*}/${*}/ -type f -name "${lang}.drs.sbn" | xargs -I{} sh -c "cat {}; echo ''" > sbn.txt
        python ~/SBN-Process/0_get_raw_file.py -i sbn.txt -lang ${lang} -ipath ./ -o all_sbn.txt
        cd -
    done
done


### 3.Split gold data to train, dev and test (need change language based on README in the official document, following example language is Italian)
raw_sbn_path=~/SBN-data/pmb-4.0.0/data/it
bash_dir=~/SBN-data/it
mkdir -p $bash_dir/gold
python ~/SBN-Process/1_split_sbn_accuracy.py -i $raw_sbn_path/gold/all_sbn.txt -ipath $raw_sbn_path/gold -o1 $bash_dir/gold/train.txt -o2 $bash_dir/gold/dev.txt -o3 $bash_dir/gold、test.txt

mkdir -p $bash_dir/silver
mkdir -p $bash_dir/bronze
cp  $raw_sbn_path/silver/all_sbn.txt $bash_dir/silver/train.txt
cp  $raw_sbn_path/silver/all_sbn.txt.raw $bash_dir/silver/train.txt.raw
cp  $raw_sbn_path/bronze/all_sbn.txt $bash_dir/bronze/train.txt
cp  $raw_sbn_path/bronze/all_sbn.txt.raw $bash_dir/bronze/train.txt.raw


### 4. Merge the data to one file, such as gold_silver, gold_silver_bronze (depending on what you need)
mkdir -p $bash_dir/gold_silver_bronze
paste $bash_dir/gold/train.txt $bash_dir/silver/train.txt $bash_dir/bronze/train.txt > $bash_dir/gold_silver_bronze/train.txt
paste $bash_dir/gold/train.txt.raw $bash_dir/silver/train.txt.raw $bash_dir/bronze/train.txt.raw > $bash_dir/gold_silver_bronze/train.txt.raw
cp $bash_dir/gold/dev.*  $bash_dir/gold_silver_bronze/
cp $bash_dir/gold/test.*  $bash_dir/gold_silver_bronze/


####----------------Now you have the data you need for experiments in the gold_silver_bronze file----------------

### 5. Process clause-SBN to linearized-SBN (for seq2seq models)
DATA_DIR=$bash_dir/gold_silver_bronze/
python sbn_preprocess.py  -input_src ${DATA_DIR}/train.txt -input_tgt ${DATA_DIR}/train.txt.raw -text_type seq -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt
python sbn_preprocess.py  -input_src ${DATA_DIR}/dev.txt -input_tgt ${DATA_DIR}/dev.txt.raw -text_type seq -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt
python sbn_preprocess.py  -input_src ${DATA_DIR}/test.txt -input_tgt ${DATA_DIR}/test.txt.raw -text_type seq -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt

### 6. Process clause-SBN to Graph-structure SBN (for Graph2seq models)

python sbn_preprocess.py  -input_src ${DATA_DIR}/train.txt -input_tgt ${DATA_DIR}/train.txt.raw -text_type graph -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt
python sbn_preprocess.py  -input_src ${DATA_DIR}/dev.txt -input_tgt ${DATA_DIR}/dev.txt.raw -text_type graph -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt
python sbn_preprocess.py  -input_src ${DATA_DIR}/test.txt -input_tgt ${DATA_DIR}/test.txt.raw -text_type graph -if_anony normal -if_hyper nohyper -trainfile ${DATA_DIR}/train.txt


