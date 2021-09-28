#!/bin/bash
model="scGGN"
comment="exp_eval"
hidden_dims="128 128"
latent_dim=10
max_epochs=400
patience=20
kl_frozen_rate=0.2

lr=0.001
min_kl_weight=0.1
dataset_name="Zeisel"
log_num=1
python debug_main.py --model ${model} --name ${dataset_name} --comment ${comment} --latent_dim ${latent_dim} --hidden_dims ${hidden_dims} --lr ${lr} --max_epochs ${max_epochs} --patience ${patience} --min_kl_weight ${min_kl_weight} --kl_frozen_rate ${kl_frozen_rate} --log_num ${log_num}
