THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.16' \
nohup python eacnn.py \
--actionDB 'tag_actions tag_actions1 tag_actions2 tag_actions3 tag_actions4' \
--max_text_num '64 52 33 54 111' \
--test_text_num 8 \
--epochs 20 \
--train_repeat 10 \
--batch_size 32 \
--text_name 'tb_results/tb01234_ep20.txt' \
> 505.txt & 

