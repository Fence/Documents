THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.16' \
nohup python eacnn.py \
--actionDB 'tag_actions4' \
--max_text_num '111' \
--test_text_num 11 \
--epochs 20 \
--train_repeat 10 \
--batch_size 32 \
--text_name 'tb_results/tb4_ep20.txt' \
> 504.txt & 

