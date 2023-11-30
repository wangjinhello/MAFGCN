#!/bin/bash
# Restaurants
#python eval.py
--dataset Restaurants --hidden_dim 48 --rnn_hidden 48 --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_rest.pt --num_layers 3
--dataset Restaurants --hidden_dim 48 --rnn_hidden 48 --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_rest.pt/best_model.pt --num_layers 3
#echo -e "\n"

# Tweets
#python eval.py
#--dataset Tweets --hidden_dim 51 --rnn_hidden 51 --top_k 4 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_tweet.pt --num_layers 3
#echo -e "\n"

# Laptops
#python eval.py
--dataset Laptops --hidden_dim 36 --rnn_hidden 36 --top_k 2 --head_num 3 --batch_size 8 --save_dir ./saved_models/best_model_lap.pt --num_layers 2
--dataset Laptops --hidden_dim 36 --rnn_hidden 36 --top_k 2 --head_num 3 --batch_size 8 --save_dir ./saved_models/best_model_lap.pt/best_model.pt --num_layers 2
#echo -e "finish"

#/best_model.pt