#!/bin/bash
train=true
export TZ="GMT-8"
mkdir -p logs

# Experiment variables
data='amsterdam'
seeker='knn'
model='timegan'
opt='adam'
exp="${model}_${opt}_${data}"

# # Experiment variables for testing
# model='add_noise'
# data='stock'
exp="test"

# Iteration variables
# (3800 // 128) = 29 iterations/epoch
# 20010 iterations = 690 epochs
emb_epochs=690
sup_epochs=690
gan_epochs=690

python main.py \
--exp               $exp \
--device            cuda \
--data_name         $data \
--is_train          $train \
--max_seq_len       100 \
--train_rate        0.5 \
--model_model       $model \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        128 \
--hidden_dim        20 \
--num_layers        3 \
--loss_fn           timegan \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \