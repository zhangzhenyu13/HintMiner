from programmingalpha.retrievers.relation_searcher import *
import programmingalpha
import random
import argparse
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam
from programmingalpha.models.InferenceModels import InferenceNet
from pytorch_pretrained_bert.optimization import warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.modeling import BertConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


BuildModel=InferenceNet

class SimpleTokenizer(BertTokenizer):
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[NUM]","[CODE]")
    def tokenize(self,txt):
        tokens=txt.split()
        seq_tokens=[]
        for tok in tokens:
            if tok.upper() in self.never_split:
                seq_tokens.append(tok.upper())
            else:
                seq_tokens.append(tok)

        return seq_tokens

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
def mseError(out,values):
    return np.mean(np.square(out-values))

def saveModel(model):
    output_model_file = os.path.join(args.output_dir, args.model_name+".bin")
    output_config_file = os.path.join(args.output_dir, args.model_name+".json")

    logger.info("saving model:{}".format(output_model_file))
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

def loadModel(num_labels):
    # Load a trained model and config that you have fine-tuned
    output_config_file = os.path.join(args.output_dir, args.model_name+".json")
    output_model_file = os.path.join(args.output_dir, args.model_name+".bin")

    config = BertConfig(output_config_file)
    model = BuildModel(config, num_labels=num_labels)
    logger.info("loading weights for model {} from {}".format(args.model_name,output_model_file))
    model.load_state_dict(torch.load(output_model_file))

    return model

def main():

    if not args.do_train and not args.do_eval:
        raise ValueError("train or eval? at least one need to be selected!")


    processors = {
        "semantic": SemanticPairProcessor,
    }

    num_labels_task = {
        "semantic": 4,
    }


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        from torch.cuda import set_device
        set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    else:
        logger.info("gradient_accumulation_steps {}".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        if args.overwrite==False and input("Output directory ({}) already exists and is not empty, rewrite the files?(Y/N)\n".format(args.output_dir)) not in ("Y","y"):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](dataSource)
    num_labels = num_labels_task[task_name]
    labelMap = RelationSearcher.label_map

    if args.tokenized:
        tokenizer=SimpleTokenizer(args.vocab_file,do_lower_case=args.do_lower_case,never_split=SimpleTokenizer.never_split)
    else:
        tokenizer = BertTokenizer(args.vocab_file,do_lower_case=args.do_lower_case,never_split=SimpleTokenizer.never_split)



    # train and eval
    if args.do_train:
        #configure model running parameters
        train_examples = None
        num_train_optimization_steps = None
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir)
            num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            if args.local_rank != -1:
                num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model

        model = BuildModel.from_pretrained(args.bert_model,num_labels = num_labels)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            from torch import nn
            model = nn.DataParallel(model,device_ids=device_ids)
            #from torch.cuda import set_device
            #set_device(args.gpu_ranks[0])
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        #preppare train data
        train_features = convert_examples_to_features(
            train_examples, labelMap, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)


    if args.do_eval or args.do_train:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, labelMap, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)

    def _eval_epoch(model):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        sranker=RelationSearcher(args.output_dir, args.model_name)
        s_outs=sranker.getRelationProbability(eval_dataloader)
        print("s-out",s_outs[:10])

        model.eval()
        eval_loss, eval_accuracy, eval_error = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        i=0
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            #logits=out_logits[i:i+args.eval_batch_size]
            i+=args.eval_batch_size

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1



        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        return eval_accuracy, eval_loss

    def _train_epoch(epoch_num):
        saved_at_least_one=False

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        global_step = 0
        best_eval_acc=0.0

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in range(epoch_num,):
            logger.info("Epoch_num:{}/{}".format(_,epoch_num))

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                input_ids=input_ids.to(device)
                input_mask=input_mask.to(device)
                segment_ids=segment_ids.to(device)
                label_ids=label_ids.to(device)

                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # save Model
                if (1+step)%args.eval_step_size==0:
                    eval_acc,eval_loss=_eval_epoch(model)
                    if eval_acc>best_eval_acc:
                        saveModel(model)
                        best_eval_acc=eval_acc
                        saved_at_least_one=True
                    logger.info("global step:{}, eval_accuracy:{}, eval_loss:{}, best eval_acc:{}".format(
                        global_step,eval_acc,eval_loss,best_eval_acc)
                    )

        if saved_at_least_one==False:
            saveModel(model)

    if args.do_train:
        _train_epoch(int(args.num_train_epochs))
    if args.do_eval:
        model=loadModel(num_labels)
        model.to(device)
        eval_acc,eval_loss=_eval_epoch(model)
        logger.info("eval acc and loss is {}, {}".format(eval_acc,eval_loss))

if __name__ == "__main__":
    dataSource=""

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        #required=True,
                        help="The name of the model.")
    parser.add_argument("--data_dir",
                        required=True,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--vocab_file",default=True,
                        type=str,
                        #required=True,
                        help="The name of the model.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="overwrite saved model folder")

    parser.add_argument("--bert_model", required=True, type=str,
                        #required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="semantic",
                        type=str,
                        #required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        required=True,
                        type=str,
                        #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--tokenized",action="store_true",help="is text tokenized?")

    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_step_size",
                        default=5000,
                        type=int,
                        help="eval model performance after several steps")

    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=10,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    print(args)
    device_ids=(0,1,2)
    main()
