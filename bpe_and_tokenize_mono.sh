#!/bin/bash

MAIN_PATH=$PWD
PARA_PATH="/content/gdrive/My Drive/NLP-project/para"
TOOLS_PATH=$PWD/tools
WIKI_PATH="/content/gdrive/My Drive/NLP-project/wiki"
PROCESSED_PATH=$PWD/data/processed/XLM15
CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15
FASTBPE=$TOOLS_PATH/fastBPE/fast

mkdir -p $PROCESSED_PATH


## Prepare monolingual data
# apply BPE codes and binarize the monolingual corpora
for lg in en ru zh; do
    for split in train valid test; do
    $FASTBPE applybpe $PROCESSED_PATH/$split.$lg "/content/data/wiki/$lg.$split" $CODES_PATH
    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$split.$lg
    rm -f $PROCESSED_PATH/$split.$lg
    done
    mv $PROCESSED_PATH/* "/content/gdrive/My Drive/NLP-project/processed/"
    rm -rf $PROCESSED_PATH/*
done



## Prepare parallel data
# apply BPE codes and binarize the parallel corpora
# for pair in ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh; do
#     for lg in $(echo $pair | sed -e 's/\-/ /g'); do
#         for split in train valid test; do
#             $FASTBPE applybpe $PROCESSED_PATH/$split.$pair.$lg "/content/gdrive/My Drive/NLP-project/para/$pair.$lg.$split" $CODES_PATH
#             python preprocess.py $VOCAB_PATH $PROCESSED_PATH/$split.$pair.$lg
#             rm -f $PROCESSED_PATH/$split.$pair.$lg
#         done
#     done
#     mv $PROCESSED_PATH/* "/content/gdrive/My Drive/NLP-project/processed/"
#     rm -rf $PROCESSED_PATH/*
# done

