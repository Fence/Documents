#coding:utf-8
import os
import sys
import time
import ipdb
import copy
import pickle
import pprint
import argparse
import tensorflow as tf

from utils import get_time
from Agent import Agent
from MultiAgent import MultiAgent
from EADQN import DeepQLearner
from Environment import Environment
from AFEnvironment import AFEnvironment
from ReplayMemory import ReplayMemory
from gensim.models import KeyedVectors


def args_init():
    parser = argparse.ArgumentParser()

    afenvarg = parser.add_argument_group('AFEnvironment')
    afenvarg.add_argument("--object_rate", type=float, default=0.05, help="")
    afenvarg.add_argument("--context_len", type=int, default=50, help="")

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--model_dir", type=str, default='/home/fengwf/Documents/mymodel-new-5-50', help="")
    envarg.add_argument("--words_num", type=int, default=500, help="")
    envarg.add_argument("--word_dim", type=int, default=50, help="")
    envarg.add_argument("--tag_dim", type=int, default=50, help="")
    envarg.add_argument("--nchars", type=int, default=93, help="")
    envarg.add_argument("--char_dim", type=int, default=30, help="")
    envarg.add_argument("--max_char_len", type=int, default=20, help="")
    envarg.add_argument("--char_emb_flag", type=int, default=0, help="")
    envarg.add_argument("--actionDB", type=str, default='tag_actions1', help="")
    envarg.add_argument("--max_text_num", type=str, default='10', help="")
    envarg.add_argument("--reward_assign", type=float, default=1.0, help="")
    envarg.add_argument("--action_rate", type=float, default=0.15, help="")
    envarg.add_argument("--action_label", type=int, default=2, help="")
    envarg.add_argument("--non_action_label", type=int, default=1, help="")
    envarg.add_argument("--add_obj_flag", type=int, default=1, help="")
    
    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--positive_rate", type=float, default=0.75, help="")
    memarg.add_argument("--priority", type=int, default=1, help="")
    memarg.add_argument("--save_replay", type=int, default=0, help="")
    memarg.add_argument("--load_replay", type=int, default=0, help="")
    memarg.add_argument("--save_replay_size", type=int, default=1000, help="")
    memarg.add_argument("--save_replay_name", type=str, default='saved_replay_memory.pkl', help="")
    memarg.add_argument("--time_step_batch", type=int, default=0, help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--optimizer", default='rmsprop', help="")
    netarg.add_argument("--learning_rate", type=float, default=0.0025, help="")
    netarg.add_argument("--momentum", type=float, default=0.8, help="")
    netarg.add_argument("--epsilon", type=float, default=1e-6, help="")
    netarg.add_argument("--decay_rate", type=float, default=0.88, help="")
    netarg.add_argument("--discount_rate", type=float, default=0.9, help="")
    netarg.add_argument("--batch_size", type=int, default=32, help="")
    netarg.add_argument("--batch_act_num", type=int, default=1, help="")
    netarg.add_argument("--num_k", type=int, default=2, help="")
    netarg.add_argument("--multi_cnn", type=int, default=0, help="")
    netarg.add_argument("--use_k_max_pool", type=int, default=0, help="")
    netarg.add_argument("--add_linear", type=int, default=1, help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start", type=float, default=1, help="")
    antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="")
    antarg.add_argument("--exploration_decay_steps", type=int, default=1000, help="")
    antarg.add_argument("--exploration_rate_test", type=float, default=0.0, help="")
    antarg.add_argument("--train_frequency", type=int, default=1, help="")
    antarg.add_argument("--train_repeat", type=int, default=1, help="")
    antarg.add_argument("--target_steps", type=int, default=5, help="")
    antarg.add_argument("--random_play", type=int, default=0, help="")

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--fold_id", type=int, default=0, help="")
    mainarg.add_argument("--ten_fold_valid", type=int, default=1, help="")
    mainarg.add_argument("--ten_fold_indices", type=str, default='data/cooking-10-fold-indices.pkl', help="")
    mainarg.add_argument("--result_dir", type=str, default="results/multi_agent_test", help="")
    mainarg.add_argument("--train_steps", type=int, default=0, help="")
    mainarg.add_argument("--test_one_flag", type=int, default=0, help="")
    mainarg.add_argument("--is_test", type=int, default=1, help="")
    mainarg.add_argument("--test_text_num", type=int, default=2, help="")
    mainarg.add_argument("--epochs", type=int, default=5, help="")
    mainarg.add_argument("--start_epoch", type=int, default=0, help="")
    mainarg.add_argument("--load_weights", type=str, default="", help="")
    mainarg.add_argument("--save_weights_prefix", type=str, default="", help="")
    mainarg.add_argument("--computer_id", type=int, default=1, help="")
    mainarg.add_argument("--max_replay_size", type=int, default=[100000,35000,100000], help="")
    mainarg.add_argument("--gpu_rate", type=float, default=0.24, help="")
    
    args = parser.parse_args()
    if args.load_weights:
        args.exploration_decay_steps = 1                                                        
    args.num_actions = 2*args.words_num
    if args.char_emb_flag:
        args.emb_dim = args.word_dim + args.tag_dim + args.char_dim
    else:
        args.emb_dim = args.word_dim + args.tag_dim
    if args.add_obj_flag:
        args.obj_dim = 50
        args.obj_emb_dim = args.emb_dim
        args.emb_dim += args.obj_dim
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    return args


def main(args, fold_result={}):
    start = time.time()
    print 'Current time is: %s'%get_time()
    print 'Starting at main.py...'

    #Initial environment, replay memory, deep_q_net and agent
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env_act = Environment(args)
        env_obj = AFEnvironment(args)
        net_act = DeepQLearner(args, sess, 'act')
        net_obj = DeepQLearner(args, sess, 'obj')

        temp_size = env_act.train_steps * args.epochs + env_act.valid_steps
        if temp_size > args.max_replay_size[args.computer_id]:
            temp_size = args.max_replay_size[args.computer_id]
        args.replay_size = temp_size
        args.train_steps = env_act.train_steps / args.batch_act_num
        args.valid_steps = env_act.valid_steps / args.batch_act_num
        args.test_text_num = env_act.valid_steps / args.words_num
        assert args.replay_size > 0

        mem_act = ReplayMemory(args, 'act')
        mem_obj = ReplayMemory(args, 'obj')
        #agent = Agent(env_act, mem_act, net_act, args)
        agent = MultiAgent(env_act, env_obj, mem_act, mem_obj, net_act, net_obj, args)

        pp = pprint.PrettyPrinter()
        pp.pprint(args.__dict__)

        if args.load_weights:
            print('Loading weights from %s...' % args.load_weights)
            if os.path.exists(args.load_weights):
                net_act.load_weights(args.load_weights)  #load last trained weights
            else:
                print("\n!!! load_weights '%s' doesn't exist !!!\n" % args.load_weights)


        if args.test_one_flag and args.load_weights:
            pass
        else:
            # loop over epochs
            for epoch in xrange(args.start_epoch, args.start_epoch + args.epochs):
                epoch_start = time.time()
                with open(args.result_dir + "_test"+ str(epoch) + ".txt",'w') as f1:
                    for ea in args.__dict__:
                        f1.write('{}: {}\n'.format(ea, args.__dict__[ea]))

                    if args.train_steps > 0:
                        agent.train(args.train_steps, epoch)
                        if args.save_weights_prefix:
                            filename = args.save_weights_prefix + "_%d.prm" % (epoch + 1)
                            net_act.save_weights(filename)

                    if args.is_test:
                        r, p, f = agent.test(args.valid_steps, f1)
                        if fold_result:
                            fold_result['recalls'].append(r)
                            fold_result['precisions'].append(p)
                            fold_result['f_measures'].append(f)
                            f1.write('\n{}\n'.format(fold_result))
                            f1.write('Average f1 value: %f\n' % sum(fold_result['f_measures'])/len(fold_result['f_measures']))
                    epoch_end = time.time()
                    print('Total time cost of epoch %d is: %ds' % (epoch, epoch_end - epoch_start))
                    f1.write('\nTotal time cost of epoch %d is: %ds\n' % (epoch, epoch_end - epoch_start))

        if args.save_replay:
            mem_act.save(args.save_replay_name, args.save_replay_size)
        end = time.time()
        print('Total time cost: %ds' % (end - start))
        print('Current time is: %s' % get_time())



if __name__ == '__main__':
    main(args_init())
