THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.16' \
nohup python eacnn.py \
--actionDB 'tag_actions' \
--max_text_num '35' \
--test_text_num 7 \
--epochs 20 \
--train_repeat 10 \
--batch_size 32 \
--text_name 'tb_results/tb0_ep20_text35_7.txt' \
> 507.txt 2>&1 & 

