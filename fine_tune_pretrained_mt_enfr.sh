# Train pretrained model more
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
