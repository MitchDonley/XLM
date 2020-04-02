# Train pretrained model more

<<<<<<< HEAD
python train.py

## main parameters
--exp_name test_enfr_mlm_tlm            # experiment name
--dump_path ./dumped/                   # where to store the experiment
-reload_model 'mlm_enfr_1024.pth'       # model to reload for encoder,decoder

## data location / training objective
--data_path ./data/processed/en-fr/     # data location
--lgs 'en-fr'                           # considered languages
--clm_steps ''                          # CLM objective
--mlm_steps 'en,fr,en-fr'               # MLM, TLM objective

## transformer parameters
--emb_dim 1024                          # embeddings / model dimension
--n_layers 6                            # number of layers
--n_heads 8                             # number of heads
--dropout 0.1                           # dropout
--attention_dropout 0.1                 # attention dropout
--gelu_activation true                  # GELU instead of ReLU

## optimization
--batch_size 32                         # sequences per batch
--bptt 256                              # sequences length
--optimizer adam,lr=0.0001              # optimizer
--epoch_size 200000                     # number of sentences per epoch
--validation_metrics _valid_mlm_ppl     # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,10  # end experiment if stopping criterion does not improve

=======
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
>>>>>>> d5c47e5996e5f46fd3470dbb8d46b6a6aa10958a
