#!/bin/bash

python glue-xnli.py \
--exp_name test_xnli_mlm_tlm \
--dump_path "/content/gdrive/My Drive/NLP-project/dumped/" \
--model_path mlm_tlm_xnli15_1024.pth \
--data_path "/content/gdrive/My Drive/NLP-project/processed" \
--transfer_tasks XNLI \
--dropout 0.01 \
--finetune_layers "0:_1" \
--batch_size 4 \
--n_epochs 250 \
--epoch_size 20000 \
--max_len 256 \
--max_vocab 95000 \