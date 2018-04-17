for fi in 1 2 3 4 5 6 7 8 9
do   
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --fold_id $fi \
        --computer_id 1 \
        --reward_assign 1.0 \
        --multi_cnn 0 \
        --use_k_max_pool 0 \
        --num_k 2 \
        --add_linear 1 \
        --epochs 5 \
        --positive_rate 0.75 \
        --load_weights '' \
        --save_weights_prefix '' \
        --gpu_rate 0.24 \
        --result_dir "results/cooking_fold$fi"
done
