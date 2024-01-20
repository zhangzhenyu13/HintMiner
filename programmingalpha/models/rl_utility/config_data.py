#training parameters
batch_size=1

train_steps=100000
warmup_steps=6000
init_lr=1e-5
max_train_epoch = 100
display_steps = 100
eval_steps = 500
teacher_forcing_rate=0.
max_seq_length=512

#pickle data config
filename_prefix = "processed."
input_dir = '/home/LAB/zhangzy/ProjectData/texarData/data/seq2seq/'
vocab_file = input_dir + '/processed.vocab.pickle'
pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 100


#text data config
vocab='/home/LAB/zhangzy/ProjectData/texarData/vocab.txt'
prefix="/home/LAB/zhangzy/ProjectData/seq2seq/"

train = {
    'num_epochs': max_train_epoch,
    'batch_size': batch_size,
    'allow_smaller_final_batch': False,
    "shuffle":True,
    'source_dataset': {
        "files": prefix+'train-src',
        'vocab_file': vocab,
        'max_seq_length': max_seq_length,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    },
    'target_dataset': {
        'files': prefix+'train-dst',
        'vocab_share': True,
        'max_seq_length': max_seq_length,
        "processing_share":True,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
}

val = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": prefix+'valid-src',
        'vocab_file': vocab,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    },
    'target_dataset': {
        'files': prefix+'valid-dst',
        'vocab_share': True,
        "processing_share":True,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
}

test = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": prefix+'valid-src',
        'vocab_share': True,
        "processing_share":True,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    },
    'target_dataset': {
        'files': prefix+'valid-dst',
        'vocab_share': True,
        "processing_share":True,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
}
