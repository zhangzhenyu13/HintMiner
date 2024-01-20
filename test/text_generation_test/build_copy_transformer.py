from programmingalpha.models.TextGenModels import TextGeneratorModel
from pytorch_pretrained_bert import optimization as bertOptimizer
import numpy as np
import torch
from torch import nn
import random
import onmt
import argparse
import programmingalpha
import logging
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts

#from programmingalpha.models.openNMT_utils import paralleltrainer
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def trainModel(textGen:TextGeneratorModel,opt,vocab_fields,train_data_files=None,valid_data_files=None):
    model=textGen.transformer
    #init random state
    #define some hyper-paprameters
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    n_gpu=len(opt.gpu_ranks)

    device = "cuda" if n_gpu>0 and torch.cuda.is_available() else "cpu"

    gpu_rank=0

    if n_gpu>1:
        model=nn.DataParallel(model)
        n_gpu=1

    if  n_gpu<1:
        gpu_rank=-1
    else:
        gpu_rank=opt.gpu_ranks[0]

    logger.info("gpu rank={}, gpu num={}, device={}".format(gpu_rank,n_gpu,device))

    torch.cuda.set_device(gpu_rank)
    model.to(torch.device(device))

    #fields data

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    unk_id=tgt_vocab.stoi[tgt_text_field.unk_token]
    #define loss
    loss=onmt.modules.CopyGeneratorLossCompute(
        criterion=onmt.modules.CopyGeneratorLoss(vocab_size=len(tgt_vocab), force_copy=False,
                    unk_index=unk_id,ignore_index=tgt_padding, eps=1e-20),
        generator=(model.module if hasattr(model, 'module') else model).generator,
        tgt_vocab=tgt_vocab, normalize_by_length=True
    )

    #configure optimizer
    lr = opt.learning_rate

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

    warmup_proportion=opt.warmup_steps/opt.train_steps
    bert_optimizer = bertOptimizer.BertAdam(params=optimizer_grouped_parameters, lr=lr, warmup=warmup_proportion,
                                            t_total=opt.train_steps
                                            )

    optim = onmt.utils.optimizers.Optimizer(
        bert_optimizer, learning_rate=lr, max_grad_norm=2)

    logger.info(model)

    train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=train_data_files,
                                                         fields=vocab_fields,
                                                         batch_size=opt.batch_size,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True)

    valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=valid_data_files,
                                                         fields=vocab_fields,
                                                         batch_size=opt.batch_size,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=False,
                                                         repeat=False)

    report_manager = onmt.utils.ReportMgr(
        report_every=opt.valid_steps, start_time=-1, tensorboard_writer=None)

    saver=onmt.models.ModelSaver(base_path=opt.save_model,
                                 model=model.module if hasattr(model, 'module') else model,
                                 model_opt=opt,
                                 fields=vocab_fields,
                                 optim=optim,keep_checkpoint=opt.keep_checkpoint)

    trainer = onmt.Trainer(model=model,
                           train_loss=loss,
                           valid_loss=loss,
                           optim=optim,shard_size=32,grad_accum_count=opt.accum_count,
                           report_manager=report_manager,
                           model_saver=saver,n_gpu=n_gpu,gpu_rank=gpu_rank,
                           model_dtype=opt.model_type)

    trainer.train(train_iter=train_iter,
                  train_steps=opt.train_steps,
                  valid_iter=valid_iter,
                  valid_steps=opt.valid_steps,
                  save_checkpoint_steps=opt.save_checkpoint_steps)

def runTrain(opt):
    import os
    data_dir=opt.data
    data_dir=data_dir[:1+data_dir.rfind("/")]
    train_data_files=[]
    validate_data_files=[]

    for filename in os.listdir(data_dir):
        if "train" in filename:
            train_data_files.append(os.path.join(data_dir,filename))
        elif "valid" in filename:
            validate_data_files.append(os.path.join(data_dir,filename))

    logger.info("train files:{}".format(train_data_files))
    logger.info("validate files:{}".format(validate_data_files))

    vocab_data=opt.data+".vocab.pt"
    logger.info("loading vocab from {}".format(vocab_data))
    vocab_fields=torch.load(vocab_data)

    TextGeneratorModel.model_opt=opt
    TextGeneratorModel.opt=opt
    TextGeneratorModel.fields=vocab_fields
    textGen=TextGeneratorModel()

    trainModel(textGen,opt,vocab_fields,train_data_files=train_data_files,valid_data_files=validate_data_files)

def generateText(textGen:TextGeneratorModel,test_data_file,opt):
    from onmt import translate
    from onmt import inputters

    src_reader = onmt.inputters.str2reader["text"]
    tgt_reader = onmt.inputters.str2reader["text"]
    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                             beta=0.,
                                             length_penalty="avg",
                                             coverage_penalty="none")

    device = "cuda" if opt.gpu_ranks>0 and torch.cuda.is_available() else "cpu"

    model=textGen.transformer
    vocab_fields=torch.load(opt.data+"/vocab.pt")
    batch_size=4
    gpu = opt.gpu_ranks[0] if device=="cuda" else -1

    model.to(torch.device(device))

    translator = translate.Translator(model=model,
                                           fields=vocab_fields,
                                           src_reader=src_reader,
                                           tgt_reader=tgt_reader,
                                           global_scorer=scorer,
                                           copy_attn=True,
                                           gpu=gpu)

    builder = translate.TranslationBuilder(data=torch.load(test_data_file),
                                                fields=vocab_fields)

    valid_iter = inputters.inputter.DatasetLazyIter(dataset_paths=[test_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=batch_size,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=False,
                                                     repeat=False)

    for batch in valid_iter:
        trans_batch = translator.translate_batch(
            batch=batch, src_vocabs=batch.dataset.src_vocabs,
            attn_debug=False)
        translations = builder.from_batch(trans_batch)
        for trans in translations:
            print(trans.log(0))

def runPrediction(opt):

    #textGen=TextGeneratorModel()
    #textGen.loadModel("/home/LAB/zhangzy/ProjectModels/knowledgeComprehension_step_6000.pt")

    # Load some data
    #validate_data_files=[ "/home/LAB/zhangzy/ProjectData/openNMT/knowledgeData.valid.0.pt" ]

    vocab_fields=torch.load("/home/LAB/zhangzy/ProjectData/openNMT/knowledgeData.vocab.pt")
    #'''
    #load vocabs
    vocab_fields = vocab_fields

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    
    print(src_text_field.pad_token,src_text_field.eos_token,src_text_field.init_token,src_text_field.unk_token)
    print(tgt_text_field.pad_token,tgt_text_field.eos_token,tgt_text_field.init_token,tgt_text_field.unk_token)

    print(type(src_vocab.stoi),type(tgt_vocab.stoi))
    count=0
    conflicts=[]
    for k in tgt_vocab.stoi:
        idx1=tgt_vocab.stoi[k]
        idx2=src_vocab.stoi[k]
        if idx1<100:
            print(k,idx1,idx2)
        try:
            assert idx1==idx2
        except:
            count+=1
            conflicts.append((k,idx1,idx2))

    print(count)
    print(conflicts)
    #'''

    #generateText(textGen,validate_data_files[0],opt)


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)


    runTrain(opt)
    #runPrediction(opt)

def _get_parser():
    parser = ArgumentParser(description='build_copy_transformer.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser
if __name__ == '__main__':

    parser = _get_parser()

    opt = parser.parse_args()
    print(opt)
    main(opt)
