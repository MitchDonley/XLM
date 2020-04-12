python -W ignore train.py \
--exp_name unsupMT_esvi_test \
--dump_path ./dumped/ \
--data_path ./data/es_vi/processed/ \
--reload_model "mlm_100_1280.pth,mlm_100_1280.pth" \
--lgs 'es-vi' \
--ae_steps 'es,vi' \
--bt_steps 'es-vi-es,vi-es-vi' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--encoder_only false \
--emb_dim 1280 \
--n_layers 16 \
--n_heads 16 \
--max_vocab 200000 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 500 \
--batch_size 8 \
--bptt 64 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 200000 \
--eval_bleu true \
--stopping_criterion 'valid_es-vi_mt_bleu,10' \
--validation_metrics 'valid_es-vi_mt_bleu'
