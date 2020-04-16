python train.py \
--exp_name fine_tune_xnli_tlm_contrastive \
--dump_path "/content/gdrive/My Drive/NLP-project/dumped/" \
--reload_model mlm_tlm_xnli15_1024.pth \
--data_path "/content/gdrive/My Drive/NLP-project/processed" \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh' \
--clm_steps '' \
--mlm_steps 'en-ar,en-bg,en-de,en-el,en-es,en-fr,en-hi,en-ru,en-sw,en-th,en-tr,en-ur,en-vi,en-zh,ar-en,bg-en,de-en,el-en,es-en,fr-en,hi-en,ru-en,sw-en,th-en,tr-en,ur-en,vi-en,zh-en' \
--emb_dim 1024 \
--n_layers 12 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--contrastive_loss true \
--max_vocab 95000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0 \
--epoch_size 200000 \
--validation_metrics _valid_mlm_ppl \
--stopping_criterion _valid_mlm_ppl,10 \