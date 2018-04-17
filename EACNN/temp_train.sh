THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.2' \
nohup python eacnn.py \
--actionDB 'tag_actions4' \
--max_text_num '61' \
--test_text_num 6 \
--epochs 100 \
--train_repeat 1 \
--batch_size 8 \
--text_name 'tb_results/tb4/tt556_result2.txt' \
> 5562.txt 2>&1 &
