#!/bin/bash
start_time=$(date +%s)
for fi in 0 1 2 3 4 5 6 7 8 9
do
    for bs in 128
    do
        python main.py \
        --ten_fold_valid 1 \
        --target_steps 150 \
        --batch_size 32  \
        --epochs 4 \
        --start_epoch 0 \
        --positive_rate 0.85 \
        --actionDB 'tag_actions tag_actions1 tag_actions2 tag_actions3 tag_actions5 tag_actions6' \
        --max_text_num '64 52 33 54 35 43' \
        --test_text_num 8 \
        --char_emb_flag 0 \
        --tag_dim 50 \
        --decay_rate 0.88 \
        --momentum 0.8 \
        --gpu_rate 0.24 \
        --load_weights "" \
        --save_weights_prefix '' \
        --result_dir "results/tenfold/fold"$fi"_bs128_or10_ts150_adr1" \
        --computer_id 1
    done
done 
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
