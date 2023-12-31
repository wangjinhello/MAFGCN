import os
import six
import torch
import random
import sys
import argparse
import pickle
import numpy as np
from utils import helper
from shutil import copyfile
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer

# fitlog log logs

# list_i=[0]
for i in range(0,51):
    # ***** Wang modified the code*****(strat)
    # Three initialization options are provided
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    # ***** Wang modified the code*****(end)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Restaurants')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--dep_dim',type=int,default=30,help='Deprel embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=300, help='GCN mem dim.')
    parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=3, help='Num of GCN layers.')
    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.4, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False)
    parser.add_argument('--loop', default=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--l2reg', type=float, default=1e-5, help='l2 .')
    parser.add_argument('--num_epoch', type=int, default=300, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--seed', type=int, default=i)#list_i[i]
    parser.add_argument('--beta', default=1.0e-04, type=float)
    parser.add_argument('--theta', default=1.0, type=float)
    # parser.add_argument('--gamma', default=1e-4, type=float)
    parser.add_argument('--head_num', default=3, type=int, help='head_num must be a multiple of 3')
    parser.add_argument('--top_k', default=2, type=int)
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--optimizer', type=str, default='Adma', help='Adma; SGD')
    parser.add_argument('--second_layer', type=str, default='nomax')
    # ***** Wang modified the code*****(start)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    # ***** Wang modified the code*****(end)
    args = parser.parse_args()

    # ***** Wang modified the code*****(start)
    args.initializer = initializers[args.initializer]
    # ***** Wang modified the code*****(end)

    # if you want to reproduce the result, fix the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    helper.print_arguments(args)

    # load contants
    dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
    vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
    token_vocab = dict()
    with open(vocab_file, 'rb') as infile:
        token_vocab['i2w'] = pickle.load(infile)
        token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}

    emb_file = './dataset/'+args.dataset+'/embedding.npy'
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == len(token_vocab['i2w'])
    assert emb_matrix.shape[1] == args.emb_dim

    args.token_vocab_size = len(token_vocab['i2w'])
    args.post_vocab_size = len(dicts['post'])
    args.pos_vocab_size = len(dicts['pos'])
    args.dep_vocab_size = len(dicts['dep'])

    dicts['token'] = token_vocab['w2i']

    # load training set and test set
    print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
    train_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/train.json', args.batch_size, args, dicts)]
    test_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)]

    # create the folder for saving the best models and log file
    model_save_dir = args.save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="#poch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")

    trainer = GCNTrainer(args, emb_matrix=emb_matrix)

    # ################################
    train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
    test_acc_history = [0.]
    for epoch in range(1, args.num_epoch+1):
        print('\nepoch:%d' %epoch)
        train_loss, train_acc, train_step = 0., 0., 0
        for batch in train_batch:
            loss, acc = trainer.update(batch)
            train_loss += loss
            train_acc += acc
            train_step += 1
            # if (epoch==1 or epoch>=?) and (train_step % args.log_step == 0):
            if train_step % args.log_step == 0:
                print("train_loss: {:1.4f}, train_acc: {:1.4f}".format(train_loss/train_step, train_acc/train_step))
        # eval on test
        # print("Evaluating on test set...")
        predictions, labels = [], []
        test_loss, test_acc, test_step = 0., 0., 0
        for batch1 in test_batch:
            loss, acc, pred, label, _, _ = trainer.predict(batch1)
            test_loss += loss
            test_acc += acc
            predictions += pred
            labels += label
            test_step += 1
        # f1 score
        f1_score = metrics.f1_score(labels, predictions, average='macro')

        print("trian_loss: {:1.4f}, test_loss: {:1.4f}, train_acc: {:1.4f}, test_acc: {:1.4f}, "
              "f1_score: {:1.4f}".format(
            train_loss/train_step, test_loss/test_step,
            train_acc/train_step, test_acc/test_step,
            f1_score))

        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
            epoch, train_loss/train_step, test_loss/test_step,
            train_acc/train_step, test_acc/test_step,
            f1_score))

        train_acc_history.append(train_acc/train_step)
        train_loss_history.append(train_loss/train_step)
        test_loss_history.append(test_loss/test_step)

        # save best model
        if (epoch == 1 and batch==0) or test_acc/test_step > max(test_acc_history):
            trainer.save(model_save_dir + '/best_model.pt')
            # print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"
                .format(epoch, train_loss/train_step, test_loss/test_step,
                train_acc/train_step, test_acc/test_step,
                f1_score))

        test_acc_history.append(test_acc/test_step)
        f1_score_history.append(f1_score)

    print("Training ended with {} epochs.".format(epoch))
    bt_test_acc = max(test_acc_history)
    bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
    bt_f1_score1 = max(f1_score_history)
    print("best test_acc/f1_score: {}/{}".format(bt_test_acc, bt_f1_score))
    with open('result.txt','a') as f:
        f.write('seed={} ->> best model: {}/{}\n'.format(i,bt_test_acc, bt_f1_score))#list_i[i]
        # f.write('best test_acc/f1_score: {}/{}\n'.format(bt_test_acc, bt_f1_score1))
        f.write('epoch={}\n'.format(test_acc_history.index(bt_test_acc)))