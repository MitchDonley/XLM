#/bin/bash

python translate.py \
--exp_name translate_en_fr \
--exp_id 0 \
--batch_size 16 \
--model_path ./dumped/unsup_MT_model/best-valid_en-fr_mt_bleu.pth \
--output_path ./data/para/syn.en-fr.fr.train \
--source_path ./data/para/en-fr.en.train \
--src_lang en \
--tgt_lang fr \