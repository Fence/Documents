THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.16' \
nohup python eacnn.py \
--actionDB 'tag_actions3' \
--max_text_num '54' \
--test_text_num 8 \
--epochs 20 \
--train_repeat 10 \
--batch_size 32 \
--text_name 'tb_results/tb3_ep20.txt' \
> 503.txt & 

