#!/bin/bash

python glue-xnli.py \
--exp_name test_xnli_tlm_fine_tune \
--dump_path "/content/gdrive/My Drive/NLP-project/dumped/" \
--model_path ./best-avg_valid_tlm_ppl.pth \
--data_path "/content/gdrive/My Drive/NLP-project/processed" \
--transfer_tasks XNLI \
--optimizer_e adam,lr=0.00004 \
--optimizer_p adam,lr=0.00004 \
--finetune_layers "0:_1" \
--batch_size 8 \
--n_epochs 250 \
--epoch_size 10000 \
--max_len 256 \
--max_vocab 95000 \