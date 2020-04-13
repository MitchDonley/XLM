#/bin/bash

python translate.py \
--exp_name translate_en_fr \
--exp_id 0 \
--batch_size 128 \
--data_size 500000 \
--model_path ./dumped/unsup_MT_model/best-valid_en-fr_mt_bleu.pth \
--output_path ./data/para/syn.en-fr.fr.train \
--src_path ./data/para/en-fr.en.train \
--src_lang en \
--tgt_lang fr \
