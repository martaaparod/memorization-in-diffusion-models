#!/bin/bash

insert_dt=0.01
insert_m=4
insert_v=6
insert_train=20000
insert_zipf_exponent=1
insert_zipf_layer=1
insert_tuple_size=2
insert_num_layers=4

insert_width=4096
insert_n_trajectories=32
insert_batch_size=32
insert_n_epoch=55000


#source ../rhm_env/bin/activate

python main.py --dataset rhm --process discrete --tuple_size ${insert_tuple_size} --num_layers ${insert_num_layers} --seed_rules 1951 --test_size 1024 --n_trajectories ${insert_n_trajectories} --model bpUnet --model_type start --model_output logits --seed_model 2 --print_period 100 --n_epoch ${insert_n_epoch} --nT 200 --optim adam --batch_size ${insert_batch_size} --lr ${insert_dt} --seed_sample 1 --save_freq 1 --num_features ${insert_v} --num_classes ${insert_v} --num_synonyms ${insert_m} --train_size ${insert_train} --output bpUnet-L${insert_num_layers}_s${insert_tuple_size}/width${insert_width}_v${insert_v}_m${insert_m}_P${insert_train}-adam_lr${insert_dt} --replacement --zipf_layer ${insert_zipf_layer} --zipf_exponent ${insert_zipf_exponent} --width ${insert_width} --unique > output.log 2> error.log &