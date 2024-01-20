import argparse

import tensorflow as tf
import opennmt as onmt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_dir", default="model",
                    help="Checkpoint directory.")

# Step 1
parser.add_argument("--src", required=True, help="Source file.")
parser.add_argument("--tgt", required=True, help="Target file.")
parser.add_argument("--vocab", required=True, help="Target vocabulary.")
parser.add_argument("--direction", required=True, type=int,
                    help="1 = translation source, 2 = translate target.")

args = parser.parse_args()


# Step 1

def load_data(input_file, input_vocab):
  """Returns an iterator over the input file.
  Args:
    input_file: The input text file.
    input_vocab: The input vocabulary.
  Returns:
    A dataset batch iterator.
  """
  dataset = tf.data.TextLineDataset(input_file)
  dataset = dataset.map(lambda x: tf.string_split([x]).values)
  dataset = dataset.map(input_vocab.lookup)
  dataset = dataset.map(lambda x: {
      "ids": x,
      "length": tf.shape(x)[0]})
  dataset = dataset.padded_batch(64, {
      "ids": [None],
      "length": []})
  return dataset.make_initializable_iterator()

if args.direction == 1:
  src_file, tgt_file = args.src, args.tgt
  src_vocab_file, tgt_vocab_file = args.src_vocab, args.tgt_vocab
else:
  src_file, tgt_file = args.tgt, args.src
  src_vocab_file, tgt_vocab_file = args.tgt_vocab, args.src_vocab

from opennmt.utils.misc import count_lines

tgt_vocab_size = count_lines(tgt_vocab_file) + 1
src_vocab_size = count_lines(src_vocab_file) + 1
from tensorflow.contrib import lookup
src_vocab = lookup.index_table_from_file(
    src_vocab_file,
    vocab_size=src_vocab_size - 1,
    num_oov_buckets=1)

with tf.device("cpu:0"):
  src_iterator = load_data(src_file, src_vocab)

src = src_iterator.get_next()


# Step 2


hidden_size = 768
from bert.modeling import BertModel,BertConfig
bert_config=BertConfig.from_json_file("")
class BertEncoder(object):
    def __init__(self,config, is_training,input_ids,input_mask=None,token_type_ids=None):
        self.model = BertModel(
          config=config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=token_type_ids)

        self.embeddings_table=self.model.get_embedding_table()

    def encode(self):
        #encoded is => sequence_output` shape = [batch_size, seq_length, hidden_size].
        output=self.model.get_sequence_output()
        states=()
        for layer in self.model.get_all_encoder_layers():
            states+=(tf.reduce_mean(layer,axis=1),)
        return output,states,


from opennmt.decoders.self_attention_decoder import _SelfAttentionDecoderLayer
from opennmt.decoders import decoder
from opennmt.layers import common, transformer
from opennmt.layers import SinusoidalPositionEncoder

class SelfAttentionDecoderV2(decoder.DecoderV2):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.

  Note:
    TensorFlow 2.0 version.
  """

  def __init__(self,
               num_layers,
               num_units=768,
               num_heads=12,
               ffn_inner_dim=3072,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               embedding_table=None,
               position_encoder=SinusoidalPositionEncoder(),
               num_sources=1,
               **kwargs):

    """Initializes the parameters of the decoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      ffn_dropout: The probability to drop units from the activation output in
        the feed forward layer.
      ffn_activation: The activation function to apply between the two linear
        transformations of the feed forward layer.
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs.
      num_sources: The number of source contexts expected by this decoder.
      **kwargs: Additional layer arguments.
    """
    super(SelfAttentionDecoderV2, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.embedding_table=embedding_table
    self.position_encoder = position_encoder

    self.layer_norm = common.LayerNorm(name="output_norm")
    self.layers = [
        _SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            name="layer_%d" % i)
        for i in range(num_layers)]

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if sequence_length is not None:
      mask = transformer.build_future_mask(
          sequence_length, maximum_length=tf.shape(inputs)[1])

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = []
      for mem, mem_length in zip(memory, memory_sequence_length):
        mem_mask = tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1], dtype=tf.float32)
        mem_mask = tf.expand_dims(mem_mask, 1)
        memory_mask.append(mem_mask)

    # Run each layer.
    new_cache = []
    for i, layer in enumerate(self.layers):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              training=None):
    _ = initial_state
    return self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    inputs = tf.expand_dims(inputs, 1)
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

#enc-dec
transformer_encoder=BertEncoder(config=bert_config,is_training=True,input_ids=src["ids"])
transformer_decoder=SelfAttentionDecoderV2(num_layers=4)

encoder = onmt.encoders.BidirectionalRNNEncoder(2, hidden_size)
decoder = onmt.decoders.AttentionalRNNDecoder(
    2, hidden_size, bridge=onmt.layers.CopyBridge())

with tf.variable_scope("src" if args.direction == 1 else "tgt"):
  src_emb = tf.get_variable("embedding", shape=[src_vocab_size, 300])
  src_gen = tf.layers.Dense(src_vocab_size)
  src_gen.build([None, hidden_size])

with tf.variable_scope("tgt" if args.direction == 1 else "src"):
  tgt_emb = tf.get_variable("embedding", shape=[tgt_vocab_size, 300])
  tgt_gen = tf.layers.Dense(tgt_vocab_size)
  tgt_gen.build([None, hidden_size])


# Step 3


from opennmt import constants

def encode():
  """Encodes src.
  Returns:
    A tuple (encoder output, encoder state, sequence length).
  """
  with tf.variable_scope("encoder"):
    return encoder.encode(
        tf.nn.embedding_lookup(src_emb, src["ids"]),
        sequence_length=src["length"],
        mode=tf.estimator.ModeKeys.PREDICT)

def decode(encoder_output):
  """Dynamically decodes from the encoder output.
  Args:
    encoder_output: The output of encode().
  Returns:
    A tuple with: the decoded word ids and the length of each decoded sequence.
  """
  batch_size = tf.shape(src["length"])[0]
  start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
  end_token = constants.END_OF_SENTENCE_ID

  with tf.variable_scope("decoder"):
    sampled_ids, _, sampled_length, _ = decoder.dynamic_decode_and_search(
        tgt_emb,
        start_tokens,
        end_token,
        vocab_size=tgt_vocab_size,
        initial_state=encoder_output[1],
        beam_width=5,
        maximum_iterations=200,
        output_layer=tgt_gen,
        mode=tf.estimator.ModeKeys.PREDICT,
        memory=encoder_output[0],
        memory_sequence_length=encoder_output[2])
    return sampled_ids, sampled_length

encoder_output = encode()
sampled_ids, sampled_length = decode(encoder_output)

tgt_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
  tgt_vocab_file,
  vocab_size=tgt_vocab_size - 1,
  default_value=constants.UNKNOWN_TOKEN)

tokens = tgt_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
length = sampled_length


# Step 4


from opennmt.utils.misc import print_bytes

saver = tf.train.Saver()
checkpoint_path = tf.train.latest_checkpoint(args.model_dir)

def session_init_op(_scaffold, sess):
  saver.restore(sess, checkpoint_path)
  tf.logging.info("Restored model from %s", checkpoint_path)

scaffold = tf.train.Scaffold(init_fn=session_init_op)
session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)

with tf.train.MonitoredSession(session_creator=session_creator) as sess:
  sess.run(src_iterator.initializer)
  while not sess.should_stop():
    _tokens, _length = sess.run([tokens, length])
    for b in range(_tokens.shape[0]):
      pred_toks = _tokens[b][0][:_length[b][0] - 1]
      pred_sent = b" ".join(pred_toks)
      print_bytes(pred_sent)
