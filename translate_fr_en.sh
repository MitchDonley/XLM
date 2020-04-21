#/bin/bash

python translate.py \
--exp_name translate_en_fr \
--exp_id 1 \
--batch_size 128 \
--start_idx 500000 \
--data_size 500000 \
--model_path ./dumped/unsup_MT_model/best-valid_en-fr_mt_bleu.pth \
--output_path ./data/para/syn.en-fr.en.train \
--src_lang_output_path ./data/syn/en-fr.fr.train.small \
--src_path ./data/para/en-fr.fr.train \
--src_lang fr \
--tgt_lang en \
