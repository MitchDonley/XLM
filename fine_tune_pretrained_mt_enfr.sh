# Train pretrained model more

python -W ignore train.py \
--exp_name test_enfr_mlm_tlm  \
--dump_path ./dumped/ \
--reload_model 'mlm_enfr_1024.pth' \
--data_path ./data/processed/en-fr-para \
--lgs 'en-fr' \
--clm_steps '' \
--mlm_steps 'en-fr' \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--batch_size 32 \
--bptt 256 \
--optimizer adam,lr=0.00001 \
--epoch_size 200000 \
--validation_metrics _valid_en_fr_mlm_ppl \
--stopping_criterion _valid_en_fr_mlm_ppl,10 \
