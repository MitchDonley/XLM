export NGPU=3;
python -W ignore -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name fine_tune_xnli_mlm_tlm_contrastive_max_pool \
--dump_path ./dumped/ \
--reload_model mlm_tlm_xnli15_1024.pth \
--data_path ./data/processed/ \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh' \
--clm_steps '' \
--mlm_steps 'ar,bg,de,en,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh,en-ar,en-bg,en-de,en-el,en-es,en-fr,en-hi,en-ru,en-sw,en-th,en-tr,en-ur,en-vi,en-zh,ar-en,bg-en,de-en,el-en,es-en,fr-en,hi-en,ru-en,sw-en,th-en,tr-en,ur-en,vi-en,zh-en' \
--emb_dim 1024 \
--n_layers 12 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--contrastive_loss true \
--contrastive_type max \
--max_vocab 95000 \
--batch_size 16 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0 \
--epoch_size 200000 \
--validation_metrics avg_valid_tlm_ppl \
--stopping_criterion avg_valid_tlm_ppl,10 \
