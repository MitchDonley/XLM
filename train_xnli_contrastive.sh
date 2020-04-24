#!/bin/bash

python glue-xnli.py \
--exp_name test_xnli_tlm_contrastive \
--dump_path "/content/gdrive/My Drive/NLP-project/dumped/" \
--model_path ./best-avg_valid_tlm_ppl.pth \
--data_path "/content/gdrive/My Drive/NLP-project/processed" \
--transfer_tasks XNLI \
--optimizer_e adam,lr=0.00003 \
--optimizer_p adam,lr=0.00003 \
--finetune_layers "0:_1" \
--batch_size 4 \
--n_epochs 250 \
--epoch_size 5000 \
--max_len 256 \
--max_vocab 95000 \