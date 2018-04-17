THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.16' \
nohup python eacnn.py \
--actionDB 'tag_actions1' \
--max_text_num '52' \
--test_text_num 8 \
--epochs 20 \
--train_repeat 10 \
--batch_size 32 \
--text_name 'tb_results/tb1_ep20.txt' \
> 501.txt & 

