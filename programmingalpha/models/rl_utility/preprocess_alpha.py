from __future__ import unicode_literals
import json
import os
import numpy as np
import pickle
import argparse
from io import open

from .config_data import unk_token_id

def split_sentence(s):

    words = s.split()

    return words


def open_file(path):
    """more robust open function"""
    return open(path, encoding='utf-8')

def read_file(path):
    """a generator to generate each line of file."""
    with open_file(path) as f:
        for line in f.readlines():
            words = split_sentence(line.strip())
            yield words

def getAllWords(path):
    with open(path,"r") as f:
        words=f.readlines()
    vocab=list(map(lambda s:s.strip(),words))
    return vocab

def make_array(word_id, words):
    """generate id numpy array from plain text words."""
    ids = [word_id.get(word, unk_token_id) for word in words]
    return np.array(ids, 'i')

def make_dataset(path, w2id):
    """generate dataset."""
    dataset, npy_dataset = [], []
    token_count, unknown_count = 0, 0
    for words in read_file(path):
        array = make_array(w2id, words)
        npy_dataset.append(array)
        dataset.append(words)
        token_count += array.size
        unknown_count += (array == unk_token_id).sum()
    print('# of tokens:{}'.format(token_count))
    print('# of unknown {} {:.2}'.format(unknown_count,
        100. * unknown_count / token_count))
    return dataset, npy_dataset

def get_preprocess_args():
    """Data preprocessing options."""
    class Config(): pass
    config = Config()
    parser = argparse.ArgumentParser(description='Preprocessing Options')


    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--vocab', type=str, default='/home/LAB/zhangzy/ProjectData/texarData/vocab.txt')
    parser.add_argument('--src', type=str, default='src')
    parser.add_argument('--tgt', type=str, default='dst')
    parser.add_argument('--input_dir', type=str,
        default='/home/LAB/zhangzy/ProjectData/seq2seq/', help='Input directory')
    parser.add_argument('--output_dir', type=str,
        default='/home/LAB/zhangzy/ProjectData/texarData/data/seq2seq/', help='Output directory')

    parser.add_argument('--save_data', type=str, default='processed',
        help='Output file for the prepared data')
    parser.parse_args(namespace=config)

    #keep consistent with original implementation
    #pylint:disable=attribute-defined-outside-init
    config.input = config.input_dir
    config.source_train = 'train-' + config.src
    config.target_train = 'train-' + config.tgt
    config.source_valid = 'valid-' + config.src
    config.target_valid = 'valid-' + config.tgt

    return config

if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    #pylint:disable=no-member
    # Vocab Construction
    source_path = os.path.join(args.input_dir, args.source_train)
    target_path = os.path.join(args.input_dir, args.target_train)

    all_words=getAllWords(os.path.join(args.input_dir,args.vocab))
    vocab=all_words

    print(all_words[:4])
    w2id = {word: index for index, word in enumerate(vocab)}

    # Train Dataset
    source_data, source_npy = make_dataset(source_path, w2id)
    target_data, target_npy = make_dataset(target_path, w2id)
    assert len(source_data) == len(target_data)

    max_seq_length=args.max_seq_length

    train_data = [(s[:max_seq_length], t[:max_seq_length]) for s, t in zip(source_data, target_data)
                  if s and t ]

    train_npy = [(s[:max_seq_length], t[:max_seq_length]) for s, t in zip(source_npy, target_npy)
                 if len(s) > 0 and len(t) > 0 ]
    assert len(train_data) == len(train_npy)

    # Display corpus statistics
    print("Vocab: {} with special tokens".format(len(vocab)))
    print('Training data size: %d' % len(train_data))

    # Valid Dataset
    source_path = os.path.join(args.input_dir, args.source_valid)
    source_data, source_npy = make_dataset(source_path, w2id)
    target_path = os.path.join(args.input_dir, args.target_valid)
    target_data, target_npy = make_dataset(target_path, w2id)
    assert len(source_data) == len(target_data)

    valid_data = [(s[:max_seq_length], t[:max_seq_length]) for s, t in zip(source_data, target_data)
                  if s and t]
    valid_npy = [(s[:max_seq_length], t[:max_seq_length]) for s, t in zip(source_npy, target_npy)
                 if len(s) > 0 and len(t) > 0]
    assert len(valid_data) == len(valid_npy)
    print('Dev data size: %d' % len(valid_data))


    id2w = {i: w for w, i in w2id.items()}

    # Save the dataset to numpy files
    train_src_output = os.path.join(args.output_dir, \
        args.save_data + '.train.' + args.src+ '.txt')
    train_tgt_output = os.path.join(args.output_dir, \
        args.save_data + '.train.' + args.tgt + '.txt')
    dev_src_output = os.path.join(args.output_dir, \
        args.save_data + '.dev.' + args.src+ '.txt')
    dev_tgt_output = os.path.join(args.output_dir, \
        args.save_data + '.dev.' + args.tgt+ '.txt')


    np.save(os.path.join(args.output_dir, args.save_data + '.train.npy'),
            train_npy)
    np.save(os.path.join(args.output_dir, args.save_data + '.valid.npy'),
            valid_npy)

    with open(os.path.join(args.output_dir, args.save_data + '.vocab.pickle'), 'wb')\
        as f:
        pickle.dump(id2w, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(train_src_output, 'w', encoding='utf-8') as fsrc, \
        open(train_tgt_output, 'w', encoding='utf-8') as ftgt:
        for words in train_data:
            fsrc.write('{}\n'.format(' '.join(words[0])))
            ftgt.write('{}\n'.format(' '.join(words[1])))
    with open(dev_src_output, 'w', encoding='utf-8') as fsrc, \
        open(dev_tgt_output, 'w', encoding='utf-8') as ftgt:
        for words in valid_data:
            fsrc.write('{}\n'.format(' '.join(words[0])))
            ftgt.write('{}\n'.format(' '.join(words[1])))


