import random
import os
import importlib
from torchtext import data
import tensorflow as tf
import texar as tx
import numpy as np

from programmingalpha.models.rl_utility import  utils,data_utils
from programmingalpha.models.rl_utility.config_data import eos_token_id, bos_token_id
from programmingalpha.models.RL_Transformer import BertRLTransformer
from programmingalpha.models.RL_Transformer import VocabWrapper
from bert import modeling
import argparse
import tqdm
import logging
from texar.utils import transformer_utils
from texar.modules.decoders import transformer_decoders
from programmingalpha.models.RL_Transformer import EmbedderWrapper
from bert import optimization

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Uses the best sample by beam search
best_results = {'score': 0, 'epoch': -1}

def computeScore(epoch,sess, hypotheses,references=None,step=None,mode="eval"):
    logger.info("computing score and save predicting results: {}/{}".format(len(hypotheses),len(references)))
    hypotheses_text=[]
    references_text=[]

    for i in range(len(hypotheses)):
        #hypotheses
        if not hypotheses[i]:
            hypotheses_text.append("unk")
        else:
            hypotheses_text .append( tx.utils.map_ids_to_strs(
                        hypotheses[i], vocab,strip_bos="[BOS]",strip_pad="[PAD]",
                        strip_eos="[EOS]", join=True)
                    )

        #references
        if references and len(references)>0:
            references_text.append( tx.utils.map_ids_to_strs(
                        references[i], vocab,strip_bos="[BOS]",strip_pad="[PAD]",
                        strip_eos="[EOS]", join=True)
            )

    logger.info("hypo:%d"%len(hypotheses_text))
    [logger.info(h) for h in hypotheses_text[:3]]
    logger.info("refs:%d"%len(references_text))
    [logger.info(r) for r in references_text[:3]]

    if not references:
        with open(os.path.join(FLAGS.model_dir, 'tmp.{}.{}.predict'.format(machine_host,mode)),"w") as f:
            f.writelines( map(lambda l:l+"\n",hypotheses_text) )
        return

    fname = os.path.join(FLAGS.model_dir, 'tmp.{}.{}'.format(machine_host,mode))
    tx.utils.write_paired_text(
        hypotheses_text, references_text, fname, mode='s',src_fname_suffix="predict",tgt_fname_suffix="truth")

    # Computes score
    bleu_scores=[]
    for ref, hyp in zip(references_text, hypotheses_text):
        bleu_one = tx.evals.sentence_bleu([ref], hyp, smooth=True)
        bleu_scores.append(bleu_one)

    eval_bleu = np.mean(bleu_scores)
    logger.info('epoch: {}, step: {}, eval_bleu {}'.format(epoch, step, eval_bleu))

    if eval_bleu > best_results['score']:
        logger.info('epoch: {}, best bleu: {}'.format(epoch, eval_bleu) )
        best_results['score'] = eval_bleu
        best_results['epoch'] = epoch
        model_path = os.path.join(FLAGS.model_dir, model_name)
        logger.info('saving model to %s' % model_path)
        BertRLTransformer.saveModel(sess,model_name)


def testModel(epoch,src_data,tgt_data=None):
    
    references, hypotheses = [], []
    bsize = config_data.batch_size
    
    beam_width = config_model.beam_width
    encoder_input, predictions=BertRLTransformer.createInferenceModel()
    beam_search_ids = predictions['sample_id'][:, :, 0]
    # Uses the best sample by beam search
    logger.info("evaluating epoch:{} with beam size={}".format(epoch,beam_width))

    with tf.Session() as sess:
        logger.info("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.loadModel(sess,model_name)
        
        #computing evaluation output
        for i in tqdm.trange(0, len(src_data), bsize):

            sources = src_data[i:i+bsize]
            if tgt_data is not None:
                targets = tgt_data[i:i+bsize]
            
            x_block = data_utils.source_pad_concat_convert(sources)
            feed_dict = {
                encoder_input: x_block,
            }
                
            fetches = {
                'beam_search_ids': beam_search_ids,
            }
            
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
    
            hypotheses.extend(h.tolist() for h in fetches_['beam_search_ids'])
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            if tgt_data is not None:
                references.extend(r.tolist() for r in targets)
                references = utils.list_strip_eos(references, eos_token_id)

        logger.info("get {} h and {} ref".format(len(hypotheses),len(references)))
        computeScore(epoch,sess,hypotheses,references,mode="test")


def train_model():
    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    labels = tf.placeholder(tf.int64, shape=(None, None))
    is_target = tf.to_float(tf.not_equal(labels, 0))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

    vocab_size=BertRLTransformer.bert_config.vocab_size

    encoder_output, embedder=BertRLTransformer.bertTransformerEncoder(True,encoder_input)
    tgt_embedding=embedder
    def __computeEmbedding(embedding_table,input_ids):

        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        flat_decoder_input_ids = tf.reshape(input_ids, [-1])
        embedded = tf.gather(embedding_table, flat_decoder_input_ids)
        input_shape = modeling.get_shape_list(input_ids)
        embedded = tf.reshape(embedded,
                              input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

        return embedded

    decoder_emb_input=__computeEmbedding(embedder,decoder_input)

    decoder = transformer_decoders.TransformerDecoder(embedding=tgt_embedding,
                             hparams=BertRLTransformer.config_model.decoder)

    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        inputs=decoder_emb_input, #embedder(decoder_input),
        sequence_length=decoder_input_length,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    #test accuracy
    accuracy=tx.evals.accuracy(labels=labels,preds=outputs.sample_id)

    mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, BertRLTransformer.config_model.loss_label_confidence)
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=BertRLTransformer.config_model.opt)


    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    tf.summary.scalar("accuracy",accuracy)
    summary_merged = tf.summary.merge_all()

    logger.info("transformer graph defined !")

    class AccurayPerformanceRecord:
        best_acc=0
        step=0

    def _eval(epoch, sess:tf.Session):
        logger.info("evaluating")
        bsize=config_data.batch_size
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])
            '''feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: 0.1,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "sample_ids":predictions
            }'''
            handle=sess.partial_run_setup([accuracy,mle_loss],[encoder_input,decoder_input,labels,tx.global_mode()])

            acc,loss = sess.partial_run(handle,fetches=[accuracy,mle_loss],feed_dict=
                {encoder_input:in_arrays[0],decoder_input:in_arrays[1],labels:in_arrays[2],tx.global_mode(): tf.estimator.ModeKeys.EVAL})

            if acc>AccurayPerformanceRecord.best_acc:
                BertRLTransformer.saveModel(sess,model_name)
                AccurayPerformanceRecord.best_acc=acc

            AccurayPerformanceRecord.step=step

            logger.info("test=> epoch:{}, acc/best_acc:{}/{}, loss:{}".format(epoch,acc,AccurayPerformanceRecord.best_acc,loss))


    #begin train or eval
    def _train_epoch(sess, epoch, step, smry_writer):
        logger.info("training epoch:{}".format(epoch))

        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            random_shuffler=data.iterator.RandomShuffler()
        )

        for train_batch in tqdm.tqdm(train_iter,desc="training"):
            #logger.info("batch",len(train_batch),)
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            #logger.info(in_arrays[0].shape,in_arrays[1].shape,in_arrays[2].shape)

            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr),
                #tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }

            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': mle_loss,
                'acc':accuracy
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            step, loss, acc = fetches_['step'], fetches_['loss'], fetches_["acc"]

            if step and step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f, acc: %.4f'%( step, loss, acc ) )
                smry_writer.add_summary(fetches_['smry'], global_step=step)

            if step and step%config_data.eval_steps==0:
                _eval(epoch,sess)


        return step

    # Run the graph
    with tf.Session() as sess:
        logger.info("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        if FLAGS.train_from :
            BertRLTransformer.loadModel(sess,model_name)
        else:
            BertRLTransformer.initBert(sess)

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        logger.info('Begin running with train_and_evaluate mode')

        best_acc=0
        step = 0
        for epoch in range(config_data.max_train_epoch):
            if step>=config_data.train_steps:
                break
            step = _train_epoch(sess, epoch, step, smry_writer)

def train_rl():

    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    qvalue_inputs = tf.placeholder(dtype=tf.float32,shape=[None, None],name='qvalue_inputs')

    def transformer_rl_model(enc_x,dec_x,enc_len,dec_len):
        encoder_output, embedder=BertRLTransformer.bertTransformerEncoder(True,enc_x)
        tgt_embedding=embedder
        def __computeEmbedding(embedding_table,input_ids):

            if input_ids.shape.ndims == 2:
                input_ids = tf.expand_dims(input_ids, axis=[-1])

            flat_decoder_input_ids = tf.reshape(input_ids, [-1])
            embedded = tf.gather(embedding_table, flat_decoder_input_ids)
            input_shape = modeling.get_shape_list(input_ids)
            embedded = tf.reshape(embedded,
                                  input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

            return embedded

        decoder_emb_input=__computeEmbedding(embedder,dec_x)

        decoder = transformer_decoders.TransformerDecoder(embedding=tgt_embedding,
                                 hparams=BertRLTransformer.config_model.decoder)

        outputs = decoder(
            memory=encoder_output,
            memory_sequence_length=enc_len,
            inputs=decoder_emb_input,
            sequence_length=dec_len,
            decoding_strategy='train_greedy',
            mode=tf.estimator.ModeKeys.TRAIN
        )

        # For training
        start_tokens = tf.fill([tx.utils.get_batch_size(enc_x)],
                               bos_token_id)

        '''decoder_emb_input=__computeEmbedding(embedder,dec_x)
        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=EmbedderWrapper(embedding_table=embedder),
            start_tokens=start_tokens,
            end_token=eos_token_id,
            top_k=1,
            softmax_temperature=0.7)

        outputs, sequence_length = decoder(
                max_decoding_length=config_model.max_decoding_length,
                helper=helper,
                mode=tf.estimator.ModeKeys.TRAIN)
        '''

        outputs, sequence_length = decoder(
            memory=encoder_output,
            memory_sequence_length=enc_len,
            inputs=decoder_emb_input,
            sequence_length=dec_len,
            start_tokens=start_tokens,
            end_token=eos_token_id,
            decoding_strategy='infer_sample',
            mode=tf.estimator.ModeKeys.TRAIN
        )


        '''from programmingalpha.models.rl_utility.seq_agent import SeqPGAgent
        agent = SeqPGAgent(
            samples=outputs.sample_id,
            logits=outputs.logits,
            sequence_length=sequence_length,
            hparams=BertRLTransformer.config_model.agent)'''

        from texar.losses.pg_losses import pg_loss_with_logits
        from texar.losses.entropy import sequence_entropy_with_logits

        agent_hparams=tx.HParams(config_model.agent,None)

        loss_hparams = agent_hparams.loss
        pg_loss = pg_loss_with_logits(
            actions=outputs.sample_id,
            logits=outputs.logits,
            sequence_length=sequence_length,
            advantages=qvalue_inputs,
            batched=True,
            average_across_batch=loss_hparams.average_across_batch,
            average_across_timesteps=loss_hparams.average_across_timesteps,
            sum_over_batch=loss_hparams.sum_over_batch,
            sum_over_timesteps=loss_hparams.sum_over_timesteps,
            time_major=loss_hparams.time_major)

        if agent_hparams.entropy_weight > 0:
            entropy=sequence_entropy_with_logits(
                outputs.logits,
                sequence_length=sequence_length,
                average_across_batch=loss_hparams.average_across_batch,
                average_across_timesteps=loss_hparams.average_across_timesteps,
                sum_over_batch=loss_hparams.sum_over_batch,
                sum_over_timesteps=loss_hparams.sum_over_timesteps,
                time_major=loss_hparams.time_major)

            pg_loss -= agent_hparams.entropy_weight * entropy


        return pg_loss, outputs, sequence_length

    agent_hparams=tx.HParams(config_model.agent,None)

    agent_loss,outputs, dec_out_seq_len =transformer_rl_model(encoder_input,decoder_input,encoder_input_length,decoder_input_length)
    predictions=outputs.sample_id
    tvars=tf.trainable_variables()
    grads=tf.gradients(agent_loss,tvars)
    grads,_=tf.clip_by_global_norm(grads,clip_norm=1.0)
    grads=list(zip(grads,tvars))

    #train method
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=config_data.init_lr, shape=[], dtype=tf.float32, name="lr")

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        config_data.train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.

    if config_data.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(config_data.warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = config_data.init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    #'''
    optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    train_op=optimizer.apply_gradients(grads,global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('agent_loss', agent_loss)
    summary_merged = tf.summary.merge_all()

    logger.info("parallel gpu computing graph for reinforcement learning defined!")

    from texar.losses.rewards import discount_reward

    def _eval(epoch, step, sess:tf.Session):
        logger.info("evaluating")
        hypotheses,references=[],[]

        bsize=config_data.batch_size
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])

            feed_dict = {
                encoder_input: in_arrays[0],
                #decoder_input: in_arrays[1],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "sample_ids":predictions
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            hypotheses.extend(h.tolist() for h in fetches_['sample_ids'])
            references.extend(r.tolist() for r in in_arrays[1])
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

        computeScore(epoch, sess, hypotheses,references, step)

    def _train_epoch(epoch, step):
        logger.info("training epoch:{}".format(epoch))

        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            random_shuffler=data.iterator.RandomShuffler()
        )

        #rl train
        for train_batch in tqdm.tqdm(train_iter,desc="training"):

            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)

            handle=sess.partial_run_setup(fetches=[predictions,dec_out_seq_len,global_step,agent_loss,train_op,summary_merged],
                                          feeds=[encoder_input,decoder_input,qvalue_inputs])
            fetches=sess.partial_run(handle,fetches={"samples":predictions,"dec_len":dec_out_seq_len},
                                     feed_dict={encoder_input:in_arrays[0] } )

            samples, decoder_out_length_py=fetches["samples"], fetches["dec_len"]
            sample_text = tx.utils.map_ids_to_strs(
                samples, vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)
            truth_text = tx.utils.map_ids_to_strs(
                in_arrays[1], vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)


            # Computes rewards
            reward = []
            for ref, hyp in zip(truth_text, sample_text):
                r = tx.evals.sentence_bleu([ref], hyp, smooth=True)
                reward.append(r)

            qvalues = discount_reward(
                reward,
                decoder_out_length_py,
                discount=agent_hparams.discount_factor,
                normalize=agent_hparams.normalize_reward)

            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                qvalue_inputs:qvalues,
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }

            # Samples
            fetches = {
                'step': global_step,
                'loss':agent_loss,
                'train_op':train_op,
                "sumry":summary_merged,
            }

            fetches = sess.run(fetches,feed_dict=feed_dict)

            # Displays
            step = fetches['step']
            loss=fetches["loss"]
            if step and step % config_data.display_steps == 0:
                logger.info("rl:epoch={}, step={}, loss={:.4f}, reward={:.4f}".format(
                    epoch, step, loss, np.mean(reward)))

                smry_writer.add_summary(fetches['smry'], global_step=step)

            if step and step%config_data.eval_steps==0:
                 _eval(epoch,step,sess)

        return step

    # Run the graph
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=sess_config) as sess:
        logger.info("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.initBert(sess,2)

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        logger.info('Begin running with train_and_evaluate mode')

        step = 0

        for epoch in range(config_data.max_train_epoch):
            if step>=config_data.train_steps:
                break
            step = _train_epoch( epoch, step)


#parallel
def average_gradients(tower_grads,verbose=0):
    average_grads_and_vars = []
    n_group=len(tower_grads)
    var_num=len(tower_grads[0])

    print(len(tower_grads))
    for i in range(var_num):
        grads=[]
        var=None
        for j in range(n_group):
            grad, var=tower_grads[j][i]
            if grad is not None:
                grad=tf.expand_dims(grad,axis=0)
                grads.append(grad)

        assert var is not None

        if not grads:
            grad_var=(None,var)
        else:
            grads=tf.concat(grads,axis=0)
            grad_sum=tf.reduce_mean(grads,axis=0)

            grad_var=(grad_sum,var)

        if verbose>0:
            logger.info(grad_var)

        average_grads_and_vars.append(grad_var)

    logger.info("n_group:{}, var_num:{}/{}".format(n_group,len(average_grads_and_vars),var_num))
    return average_grads_and_vars

def train_transformer_parallel():
    """Entrypoint.
    """

    from texar.modules.decoders import TransformerDecoder
    from texar.utils import transformer_utils

    def transformerModel(enc_x,dec_x, enc_len, dec_len):
        encoder_output, emb_tabel=BertRLTransformer.bertTransformerEncoder(True,enc_x)
        tgt_embedding=emb_tabel
        def __computeEmbedding(embedding_table,input_ids):

                if input_ids.shape.ndims == 2:
                    input_ids = tf.expand_dims(input_ids, axis=[-1])

                flat_decoder_input_ids = tf.reshape(input_ids, [-1])
                embedded = tf.gather(embedding_table, flat_decoder_input_ids)
                input_shape = modeling.get_shape_list(input_ids)
                embedded = tf.reshape(embedded,
                                      input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

                return embedded

        decoder_emb_input=__computeEmbedding(emb_tabel,dec_x)

        decoder = TransformerDecoder(embedding=tgt_embedding,
                                     hparams=config_model.decoder)
        # For training
        outputs = decoder(
            memory=encoder_output,
            memory_sequence_length=enc_len,
            inputs=decoder_emb_input, #embedder(decoder_input),
            sequence_length=dec_len,
            decoding_strategy='train_greedy',
            mode=tf.estimator.ModeKeys.TRAIN
        )

        return outputs

    def computeLoss(logits,labels):
        is_target = tf.to_float(tf.not_equal(labels, 0))
        mle_loss = transformer_utils.smoothing_cross_entropy(
        logits, labels, BertRLTransformer.bert_config.vocab_size, config_model.loss_label_confidence)
        mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)
        return mle_loss


    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    labels = tf.placeholder(tf.int64, shape=(None, None))


    #  '''
    '''
    optimizer=tx.core.AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
         weight_decay_rate=0.01,
         beta_1=0.9,
         beta_2=0.999,
         epsilon=1e-6,
         exclude_from_weight_decay=None,
         name="AdamWeightDecayOptimizer"
    )
    '''

    #train steps with data parallel computing graph
    tower_grads=[]
    n_gpu=FLAGS.gpu_num
    batch_size=config_data.batch_size
    mle_loss=[]
    predictions=[]
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        for i in range(n_gpu):
            with tf.device("%s:%d"%(device_name,i)):
                with tf.name_scope("tower_%d"%i):
                    enc_x=encoder_input[i*batch_size:(i+1)*batch_size]
                    enc_len=encoder_input_length[i*batch_size:(i+1)*batch_size]
                    dec_y=decoder_input[i*batch_size:(i+1)*batch_size]
                    dec_len=decoder_input_length[i*batch_size:(i+1)*batch_size]
                    dec_out=transformerModel(enc_x=enc_x,dec_x=dec_y,enc_len=enc_len,dec_len=dec_len)
                    dec_label=labels[i*batch_size:(i+1)*batch_size]

                    predictions.append(dec_out.sample_id)
                    tf.get_variable_scope().reuse_variables()
                    loss=computeLoss(dec_out.logits,dec_label)
                    mle_loss.append(loss)
                    #grads=optimizer.compute_gradients(loss=loss,var_list=tf.trainable_variables())

                    tvars=tf.trainable_variables()
                    grads=tf.gradients(loss,tvars)
                    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

                    tower_grads.append(list(zip(grads,tvars)))

    grads=average_gradients(tower_grads, verbose=2)

    mle_loss=tf.reduce_mean(tf.stack(mle_loss,axis=0),axis=0)
    predictions=tf.concat(predictions,axis=0,name="predictions")
    accuracy=tx.evals.accuracy(labels,predictions)

    #train method
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=config_data.init_lr, shape=[], dtype=tf.float32, name="lr")

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        config_data.train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.

    if config_data.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(config_data.warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = config_data.init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    #'''
    from bert import optimization
    optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    train_op=optimizer.apply_gradients(grads,global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_merged = tf.summary.merge_all()

    logger.info("parallel gpu computing graph defined!")

    # Uses the best sample by beam search

    bsize=config_data.batch_size*n_gpu

    def _eval(epoch, step, sess:tf.Session):
        logger.info("evaluating")
        accs=[]
        losses=[]
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])

            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "acc":accuracy,
                "loss":mle_loss
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
            acc, loss=fetches_["acc"], fetches_["loss"]
            accs.append(acc)
            losses.append(loss)

        acc=np.mean(accs)
        loss=np.mean(losses)
        logger.info("eval epoch:{}, step: {}, acc: {}, loss: {}".format(epoch, step, acc, loss))

    def _train_epoch(sess, epoch, step, smry_writer):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            bsize,
            key=lambda x: (len(x[0]), len(x[1])),
            #batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())
        accs=[]
        losses=[]

        for train_batch in tqdm.tqdm(train_iter):
            #logger.info("batch size: {}".format(len(train_batch)))
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr)
            }
            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': mle_loss,
                'acc':accuracy
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            step, loss, acc = fetches_['step'], fetches_['loss'], fetches_["acc"]

            accs.append(acc)
            losses.append(loss)
            #logger.info("step:{}".format(step))
            if step and step % config_data.display_steps == 0:
                logger.info('step: %d, batch_size: %d, loss: %.4f, acc:%.4f' % (step, config_data.batch_size*n_gpu,np.mean(losses),np.mean(accs)))
                smry_writer.add_summary(fetches_['smry'], global_step=step)
                accs.clear()
                losses.clear()

            if step and step% config_data.eval_steps==0:
                _eval(epoch,step, sess)

        return step

    # Run the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        logger.info("init vars!")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.initBert(sess,1)

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        logger.info('Begin running with train_and_evaluate mode')

        step = 0
        for epoch in range(config_data.max_train_epoch):
            step = _train_epoch(sess, epoch, step, smry_writer)




def train_rl_parallel():

    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    qvalue_inputs = tf.placeholder(dtype=tf.float32,shape=[None, None],name='qvalue_inputs')

    agent_hparams=tx.HParams(config_model.agent,None)

    def transformer_rl_model(enc_x,dec_x,enc_len,dec_len,q_val,agent_hparams):
        encoder_output, embedder=BertRLTransformer.bertTransformerEncoder(True,enc_x)
        tgt_embedding=embedder
        def __computeEmbedding(embedding_table,input_ids):

            if input_ids.shape.ndims == 2:
                input_ids = tf.expand_dims(input_ids, axis=[-1])

            flat_decoder_input_ids = tf.reshape(input_ids, [-1])
            embedded = tf.gather(embedding_table, flat_decoder_input_ids)
            input_shape = modeling.get_shape_list(input_ids)
            embedded = tf.reshape(embedded,
                                  input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

            return embedded

        decoder_emb_input=__computeEmbedding(embedder,dec_x)

        decoder = transformer_decoders.TransformerDecoder(embedding=tgt_embedding,
                                 hparams=BertRLTransformer.config_model.decoder)

        outputs = decoder(
            memory=encoder_output,
            memory_sequence_length=enc_len,
            inputs=decoder_emb_input,
            sequence_length=dec_len,
            decoding_strategy='train_greedy',
            mode=tf.estimator.ModeKeys.TRAIN
        )

        # For training
        start_tokens = tf.fill([tx.utils.get_batch_size(encoder_input)],
                               bos_token_id)

        '''decoder_emb_input=__computeEmbedding(embedder,dec_x)
        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=EmbedderWrapper(embedding_table=embedder),
            start_tokens=start_tokens,
            end_token=eos_token_id,
            top_k=1,
            softmax_temperature=0.7)

        outputs, sequence_length = decoder(
                max_decoding_length=config_model.max_decoding_length,
                helper=helper,
                mode=tf.estimator.ModeKeys.TRAIN)
        '''
        from tensorflow.contrib import seq2seq

        outputs, sequence_length = decoder(
            memory=encoder_output,
            memory_sequence_length=enc_len,
            inputs=decoder_emb_input,
            sequence_length=dec_len,
            start_tokens=start_tokens,
            end_token=eos_token_id,
            decoding_strategy='infer_sample',
            mode=tf.estimator.ModeKeys.TRAIN
        )

        from texar.losses.pg_losses import pg_loss_with_logits
        from texar.losses.entropy import sequence_entropy_with_logits


        loss_hparams = agent_hparams.loss
        pg_loss = pg_loss_with_logits(
            actions=outputs.sample_id,
            logits=outputs.logits,
            sequence_length=sequence_length,
            advantages=q_val,
            batched=True,
            average_across_batch=loss_hparams.average_across_batch,
            average_across_timesteps=loss_hparams.average_across_timesteps,
            sum_over_batch=loss_hparams.sum_over_batch,
            sum_over_timesteps=loss_hparams.sum_over_timesteps,
            time_major=loss_hparams.time_major)

        if agent_hparams.entropy_weight > 0:
            entropy=sequence_entropy_with_logits(
                outputs.logits,
                sequence_length=sequence_length,
                average_across_batch=loss_hparams.average_across_batch,
                average_across_timesteps=loss_hparams.average_across_timesteps,
                sum_over_batch=loss_hparams.sum_over_batch,
                sum_over_timesteps=loss_hparams.sum_over_timesteps,
                time_major=loss_hparams.time_major)

            pg_loss -= agent_hparams.entropy_weight * entropy


        return pg_loss, outputs, sequence_length

    #train steps with data parallel computing graph
    tower_grads=[]
    n_gpu=FLAGS.gpu_num
    batch_size=config_data.batch_size
    agent_loss=[]
    predictions=[]
    dec_out_seq_len=[]
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        for i in range(n_gpu):
            with tf.device("%s:%d"%(device_name,i)):
                with tf.name_scope("tower_%d"%i):
                    enc_x=encoder_input[i*batch_size:(i+1)*batch_size]
                    enc_len=encoder_input_length[i*batch_size:(i+1)*batch_size]
                    dec_y=decoder_input[i*batch_size:(i+1)*batch_size]
                    dec_len=decoder_input_length[i*batch_size:(i+1)*batch_size]
                    q_val=qvalue_inputs[i*batch_size:(i+1)*batch_size]
                    loss, dec_out, seq_len=transformer_rl_model(enc_x=enc_x,dec_x=dec_y,enc_len=enc_len,dec_len=dec_len,q_val=q_val,agent_hparams=agent_hparams)

                    agent_loss.append(loss)
                    predictions.append(dec_out.sample_id)
                    dec_out_seq_len.append(seq_len)
                    tf.get_variable_scope().reuse_variables()

                    #grads=optimizer.compute_gradients(loss=loss,var_list=tf.trainable_variables())

                    tvars=tf.trainable_variables()
                    grads=tf.gradients(loss,tvars)
                    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

                    tower_grads.append(list(zip(grads,tvars)))

    grads=average_gradients(tower_grads,verbose=2)
    agent_loss=tf.reduce_mean(tf.stack(agent_loss,axis=0),axis=0)
    predictions=tf.concat(predictions,axis=0,name="predictions")
    dec_out_seq_len=tf.concat(dec_out_seq_len,axis=0,name="decoder_output_length")
    #train method
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=config_data.init_lr, shape=[], dtype=tf.float32, name="lr")

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        config_data.train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.

    if config_data.warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(config_data.warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = config_data.init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    #'''
    optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    train_op=optimizer.apply_gradients(grads,global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('agent_loss', agent_loss)
    summary_merged = tf.summary.merge_all()

    logger.info("parallel gpu computing graph for reinforcement learning defined!")


    #'''
    optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    train_op=optimizer.apply_gradients(grads,global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('agent_loss', agent_loss)
    summary_merged = tf.summary.merge_all()

    logger.info("parallel gpu computing graph for reinforcement learning defined!")

    from texar.losses.rewards import discount_reward

    def _eval(epoch, step, sess:tf.Session):
        logger.info("evaluating")
        hypotheses,references=[],[]

        bsize=config_data.batch_size
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])

            feed_dict = {
                encoder_input: in_arrays[0],
                #decoder_input: in_arrays[1],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "sample_ids":predictions
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            hypotheses.extend(h.tolist() for h in fetches_['sample_ids'])
            references.extend(r.tolist() for r in in_arrays[1])
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

        computeScore(epoch, sess, hypotheses,references, step)

    def _train_epoch(epoch, step):
        logger.info("training epoch:{}".format(epoch))

        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            random_shuffler=data.iterator.RandomShuffler()
        )

        #rl train
        for train_batch in tqdm.tqdm(train_iter,desc="training"):

            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)

            handle=sess.partial_run_setup(fetches=[predictions,dec_out_seq_len,global_step,agent_loss,train_op,summary_merged],
                                          feeds=[encoder_input,decoder_input,qvalue_inputs])
            fetches=sess.partial_run(handle,fetches={"samples":predictions,"dec_len":dec_out_seq_len},
                                     feed_dict={encoder_input:in_arrays[0] } )

            samples, decoder_out_length_py=fetches["samples"], fetches["dec_len"]
            sample_text = tx.utils.map_ids_to_strs(
                samples, vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)
            truth_text = tx.utils.map_ids_to_strs(
                in_arrays[1], vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)


            # Computes rewards
            reward = []
            for ref, hyp in zip(truth_text, sample_text):
                r = tx.evals.sentence_bleu([ref], hyp, smooth=True)
                reward.append(r)

            qvalues = discount_reward(
                reward,
                decoder_out_length_py,
                discount=agent_hparams.discount_factor,
                normalize=agent_hparams.normalize_reward)

            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                qvalue_inputs:qvalues,
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }

            # Samples
            fetches = {
                'step': global_step,
                'loss':agent_loss,
                'train_op':train_op,
                "sumry":summary_merged,
            }

            fetches = sess.run(fetches,feed_dict=feed_dict)

            # Displays
            step = fetches['step']
            loss=fetches["loss"]
            if step and step % config_data.display_steps == 0:
                logger.info("rl:epoch={}, step={}, loss={:.4f}, reward={:.4f}".format(
                    epoch, step, loss, np.mean(reward)))

                smry_writer.add_summary(fetches['smry'], global_step=step)

            if step and step%config_data.eval_steps==0:
                 _eval(epoch,step,sess)

        return step

    # Run the graph
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=sess_config) as sess:
        logger.info("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.initBert(sess,2)

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        logger.info('Begin running with train_and_evaluate mode')

        step = 0

        for epoch in range(config_data.max_train_epoch):
            if step>=config_data.train_steps:
                break
            step = _train_epoch( epoch, step)




if __name__ == '__main__':
    flags=argparse.ArgumentParser()

    flags.add_argument("--use_rl", action="store_true",
                        help="wthether or not use reinforcement learning.")
    flags.add_argument("--model_name", default="transformer-rl",
                        help="name of the ouput model file.")
    flags.add_argument("--run_mode", default="train_and_evaluate",
                        help="Either train_and_evaluate or test.")
    flags.add_argument("--model_dir", default="/home/LAB/zhangzy/ProjectModels/rlmodel",
                        help="Directory to save the trained model and logs.")

    flags.add_argument("--bert_config", default="/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_config.json",
                        help="Directory to bert config json file.")
    flags.add_argument("--bert_ckpt", default="/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_model.ckpt",
                        help="Directory to bert model dir.")

    flags.add_argument("--train_from", action="store_true",
                        help="train from a previous ckpt.")

    flags.add_argument("--gpu_num",default=0,type=int,help="how many gpu to use")
    flags.add_argument("--device_name",default="/device:GPU",type=str,help="name prefix to gpu device")

    FLAGS = flags.parse_args()

    config_model = importlib.import_module("programmingalpha.models.rl_utility.config_model")
    config_data = importlib.import_module("programmingalpha.models.rl_utility.config_data")

    utils.set_random_seed(config_model.random_seed)

    BertRLTransformer.config_data=config_data
    BertRLTransformer.config_model=config_model
    BertRLTransformer.bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config)
    BertRLTransformer.bert_model_ckpt=FLAGS.bert_ckpt
    BertRLTransformer.transformer_model_dir=FLAGS.model_dir

    device_name=FLAGS.device_name
    #get host name of the running machine
    import socket
    machine_host=socket.gethostname()
    # Create model dir if not exists
    tx.utils.maybe_create_dir(FLAGS.model_dir)

    # Load data
    train_data, eval_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    #eval_data=eval_data[:100]
    #train_data=eval_data
    
    # Load vocab
    vocab=VocabWrapper(config_data.vocab)

    model_name=FLAGS.model_name+".ckpt"
    #FLAGS.run_mode="test"

    #from tensorflow.python.client import device_lib
    #logger.info("devices:{}".format(device_lib.list_local_devices()))

    logger.info(FLAGS)
    #exit(10)

    if FLAGS.run_mode=="train_and_evaluate":

        if FLAGS.use_rl:
            if FLAGS.gpu_num<2:
                logger.info("traditional training use rl")
                train_rl()
            else:
                logger.info("training use rl with multi-gpu({})".format(FLAGS.gpu_num))
                train_rl_parallel()
        else:
            if FLAGS.gpu_num<2:
                logger.info("traditional training method")
                train_model()
            else:
                logger.info("traditional training method with mulit-gpu({})".format(FLAGS.gpu_num))
                train_transformer_parallel()
            
    elif FLAGS.run_mode=="test":
        logger.info("run test")
        sources,targets=zip(*eval_data)
        testModel(0,sources)
    
    else:
        raise ValueError("run mode: {} =>not defined!".format(FLAGS.run_mode))
    
