# MT training with new pretrained model
# 0.5 size of tokens_per_batch, batch size, bptt 

python train.py

## main parameters
--exp_name unsupMT_enfr                                       # experiment name
--dump_path ./dumped/                                         # where to store the experiment
--reload_model 'mlm_enfr_1024.pth,mlm_enfr_1024.pth'          # model to reload for encoder,decoder

## data location / training objective
--data_path ./data/processed/en-fr/                           # data location
--lgs 'en-fr'                                                 # considered languages
--ae_steps 'en,fr'                                            # denoising auto-encoder training steps
#--bt_steps 'en-fr-en,fr-en-fr'                                # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 2000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_en-fr_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_en-fr_mt_bleu'                    # end experiment if stopping criterion does not improve