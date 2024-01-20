import tensorflow as tf
import texar as tx
from texar.modules import TransformerDecoder
from texar.utils import transformer_utils
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow

from .rl_utility.config_data import bos_token_id,eos_token_id,unk_token_id, pad_token_id
from bert import modeling
from bert import optimization
from tensorflow import gfile
import numpy as np
from collections import defaultdict
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbedderWrapper(object):
    def __init__(self,embedding_table):
        self.embedding_table=embedding_table
    def  __call__(self,input_ids):

        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        flat_decoder_input_ids = tf.reshape(input_ids, [-1])
        embedded = tf.gather(self.embedding_table, flat_decoder_input_ids)
        input_shape = modeling.get_shape_list(input_ids)
        embedded = tf.reshape(embedded,
                              input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

        return embedded

def _make_defaultdict(keys, values, default_value):
    """Creates a python defaultdict.

    Args:
        keys (list): Keys of the dictionary.
        values (list): Values correspond to keys. The two lists :attr:`keys` and
            :attr:`values` must be of the same length.
        default_value: default value returned when key is missing.

    Returns:
        defaultdict: A python `defaultdict` instance that maps keys to values.
    """
    dict_ = defaultdict(lambda: default_value)
    for k, v in zip(keys, values):
        dict_[k] = v

    return dict_

class VocabWrapper(tx.data.Vocab):
    def __init__(self,filename,pad_token="[PAD]",bos_token="[BOS]",eos_token="[EOS]",unk_token="[UNK]"):

        self._filename = filename
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token

        self._id_to_token_map, self._token_to_id_map, \
        self._id_to_token_map_py, self._token_to_id_map_py = \
            self.load(self._filename)

    def load(self, filename):
        """Loads the vocabulary from the file.

        Args:
            filename (str): Path to the vocabulary file.

        Returns:
            A tuple of TF and python mapping tables between word string and
            index, (:attr:`id_to_token_map`, :attr:`token_to_id_map`,
            :attr:`id_to_token_map_py`, :attr:`token_to_id_map_py`), where
            :attr:`id_to_token_map` and :attr:`token_to_id_map` are
            TF :tf_main:`HashTable <contrib/lookup/HashTable>` instances,
            and :attr:`id_to_token_map_py` and
            :attr:`token_to_id_map_py` are python `defaultdict` instances.
        """
        with gfile.GFile(filename) as vocab_file:
            # Converts to 'unicode' (Python 2) or 'str' (Python 3)
            vocab = list(tf.compat.as_text(line.strip()) for line in vocab_file)


        # Places _pad_token at the beginning to make sure it take index 0.
        # Must make sure this is consistent with the above line
        unk_token_idx = unk_token_id
        vocab_size = len(vocab)
        vocab_idx = np.arange(vocab_size)

        # Creates TF maps
        id_to_token_map = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                vocab_idx, vocab, key_dtype=tf.int64, value_dtype=tf.string),
            self._unk_token)

        token_to_id_map = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                vocab, vocab_idx, key_dtype=tf.string, value_dtype=tf.int64),
            unk_token_idx)

        # Creates python maps to interface with python code
        id_to_token_map_py = _make_defaultdict(
            vocab_idx, vocab, self._unk_token)
        token_to_id_map_py = _make_defaultdict(
            vocab, vocab_idx, unk_token_idx)

        logger.info("vocab size: {}/{}".format( len(token_to_id_map_py),len(id_to_token_map_py) ) )

        return id_to_token_map, token_to_id_map, \
               id_to_token_map_py, token_to_id_map_py

class BertRLTransformer(object):

    config_data=None
    config_model=None
    bert_config=None
    bert_model_ckpt=None
    transformer_model_dir=None
    saver=None

    @staticmethod
    def loadModel(sess:tf.Session,model_name,verbose=1):
        logger.info("loading model!")
        model_ckpt = os.path.join(BertRLTransformer.transformer_model_dir, model_name)

        currentVars=BertRLTransformer.getCurrentVariables()
        currentVarsMap=dict()
        if verbose>0:
            logger.info("********current vars({}) in the graph**********".format(len(currentVars)))
        for var in currentVars:
            if verbose>1:
                logger.info("{},{},{}".format(var,var.name,var.name[:-2]) )
            currentVarsMap[var.name[:-2]]=var

        chkptVars=BertRLTransformer.getCheckpointVars(model_ckpt)
        if verbose>0:
            logger.info("********chkpt vars({}) in the graph**********".format(len(chkptVars)))
        if verbose>1:
            for var in chkptVars:
                logger.info(var)

        variables_to_restore=[]
        for varname in chkptVars:
            if varname in currentVarsMap:
                variables_to_restore.append(currentVarsMap[varname])

        if verbose>0:
            logger.info("********restorbale vars({}) in the graph**********".format(len(variables_to_restore)))
        if verbose>1:
            for var in variables_to_restore:
                logger.info(var)

        saver=tf.train.Saver(variables_to_restore)
        saver.restore(sess,model_ckpt)
        logger.info("loaded model!")

    @staticmethod
    def saveModel(sess:tf.Session,model_name,verbose=1):
        logger.info("saving model!")
        if BertRLTransformer.saver is None:
            BertRLTransformer.saver=tf.train.Saver(max_to_keep=5)

        variables_to_save=BertRLTransformer.getCurrentVariables()
        if verbose>0:
            logger.info("********current vars({}) in the graph**********".format(len(variables_to_save)))
        if verbose>1:
            for var in variables_to_save:
                logger.info(var)

        model_path = os.path.join(BertRLTransformer.transformer_model_dir, model_name)
        saver=BertRLTransformer.saver
        saver.save(sess,model_path)
        logger.info("saved model!")

    @staticmethod
    def initBert(sess:tf.Session=None,verbose=1):
        logger.info("*************initing bert*****************")
        init_checkpoint=BertRLTransformer.bert_model_ckpt

        exlude=("transformer_decoder","OptimizeLoss")
        include=("bert",)
        variables_to_restore=slim.get_variables_to_restore(exclude=exlude,include=include)

        if verbose>0:
            logger.info("current bert variables num: {}".format(len(variables_to_restore)))
        if verbose>1:
            for var in variables_to_restore:
                logger.info("var name: {}".format(var.name))

        ckpt_vars=BertRLTransformer.getCheckpointVars(init_checkpoint)
        if verbose>0:
            logger.info("ckpt vars num:{}".format(len(ckpt_vars)))
        if verbose>1:
            for var in ckpt_vars:
                logger.info("var name:{}".format(var))

        restorable_vars=[]
        for var in variables_to_restore:
            if var.name[:-2] in ckpt_vars:
                restorable_vars.append(var)
        if verbose>0:
            logger.info("restorbale vars num: %d"%len(restorable_vars))
        if verbose>1:
            for var in restorable_vars:
                logger.info("var name: {}".format(var.name))

        #init variables from check point
        saver=tf.train.Saver(restorable_vars)
        saver.restore(sess,init_checkpoint)

        logger.info("initing bert finished!")


    @staticmethod
    def getCurrentVariables(include=None,exclude=None,verbose=0):
        variables=slim.get_variables_to_restore(include=include,exclude=exclude)
        if verbose>0:
            logger.info("********current vars({}) in the graph**********".format(len(variables)))
            for var in variables:
                logger.info(var)
        return variables

    @staticmethod
    def getVaribalesScope(scope,verbose=0):
        variables=slim.get_variables(scope=scope)
        if verbose>0:
            logger.info("********scope({}) vars({}) in the graph**********".format(scope,len(variables)))
            for var in variables:
                logger.info(var)

        return variables

    @staticmethod
    def getCheckpointVars(ckpt_file,verbose=0):
        reader=pywrap_tensorflow.NewCheckpointReader(ckpt_file)
        var_to_shape_map=reader.get_variable_to_shape_map()
        variables=[]

        if verbose>0:
            logger.info("********ckpt({}) vars({}) in the graph**********".format(ckpt_file,len(variables)))

        for key in var_to_shape_map:
            variables.append(key)
            if verbose>0:
                logger.info("tensor name:{}, shape:{}".format( key,var_to_shape_map[key]) )

        return sorted(variables)

    @staticmethod
    def bertTransformerEncoder(is_training,input_ids,input_mask=None,token_type_ids=None):
        logger.info("creating bert graph, is training:{}".format(is_training) )
        model = modeling.BertModel(
          config=BertRLTransformer.bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=token_type_ids)

        #encoded is => sequence_output` shape = [batch_size, seq_length, hidden_size].
        encoded=model.get_sequence_output()
        embedding_table=model.get_embedding_table()

        return encoded, embedding_table


    @staticmethod
    def createInferenceModel():
        # Build model graph
        encoder_input = tf.placeholder(tf.int64, shape=(None, None))
        # (text sequence length excluding padding)
        encoder_input_length = tf.reduce_sum(
            1 - tf.to_int32(tf.equal(encoder_input, pad_token_id)), axis=1)

        encoder_output, embedding_tabel=BertRLTransformer.bertTransformerEncoder(False,encoder_input)
        tgt_embedding=embedding_tabel

        decoder = TransformerDecoder(embedding=tgt_embedding,
                                     hparams=BertRLTransformer.config_model.decoder)

        # For inference
        start_tokens = tf.fill([tx.utils.get_batch_size(encoder_input)],
                               bos_token_id)
        predictions = decoder(
            memory=encoder_output,
            memory_sequence_length=encoder_input_length,
            beam_width=BertRLTransformer.config_model.beam_width,
            start_tokens=start_tokens,
            end_token=eos_token_id,
            max_decoding_length=BertRLTransformer.config_model.max_decoding_length,
            mode=tf.estimator.ModeKeys.PREDICT
        )

        logger.info("inference graph defined !")

        return encoder_input, predictions

