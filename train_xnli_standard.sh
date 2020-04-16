#!/bin/bash

python glue-xnli.py \
--exp_name test_xnli_mlm_tlm_contrastive \
--dump_path "/content/gdrive/My Drive/NLP-project/dumped/" \
--model_path mlm_tlm_xnli15_1024.pth \
--data_path "/content/gdrive/My Drive/NLP-project/processed" \
--transfer_tasks XNLI,SST-2 \
--optimizer_e adam,lr=0.000025 \
--optimizer_p adam,lr=0.000025 \
--finetune_layers "0:_1" \
--batch_size 8 \
--n_epochs 250 \
--epoch_size 20000 \
--max_len 256 \
--max_vocab 95000 \