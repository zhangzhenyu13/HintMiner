import torch
import torch.nn as nn
import onmt.utils
from pytorch_pretrained_bert.modeling import BertModel
import programmingalpha
import logging
import numpy as np
import math
from pytorch_pretrained_bert.modeling import BertEmbeddings

from copy import deepcopy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, max_len=1500):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        return emb

class OnmtBertEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size,embedinng_dim,max_posistion_size,drop_out,padding_idx,bertEmb:BertEmbeddings):
        super(OnmtBertEmbedding, self).__init__()
        self.word_embeddings=bertEmb.word_embeddings
        self.position_embeddings=PositionalEncoding(dim=embedinng_dim)
        self.LayerNorm = bertEmb.LayerNorm

        self.word_padding_idx=padding_idx

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.dropout = nn.Dropout(drop_out)

        #parameters
        self.max_position_size=max_posistion_size
        self.vocab_size=vocab_size
        self.type_num=2
        self.dropout_prob=drop_out
        self.embeddings_size=embedinng_dim

    def forward(self, input_ids,step=None):
        '''

        :param input_ids: (len*b*feat)
        :param step:
        :return: (len*b*dim)
        '''
        #print("input",input_ids.size())
        input_ids=input_ids.squeeze(2).transpose(0,1)
        #print("input",input_ids.size())

        words_embeddings = self.word_embeddings(input_ids)
        #print("word emb",words_embeddings.size())

        words_embeddings=words_embeddings.transpose(0,1)
        #print("word emb",words_embeddings.size())

        embeddings = self.position_embeddings(words_embeddings,step)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print("emb",embeddings.size())

        return embeddings


class OnmtBertTransformerEncoder(BertModel):

    def setEmbeddings(self,emb):
        self.embeddings=emb

    def forward(self, input_ids, lengths=None, attention_mask=None, output_all_encoded_layers=False):
        #print("before input ids",input_ids.size())
        #input_ids=input_ids.transpose(0,1).squeeze(2)
        #print("after input ids",input_ids.size())
        #print("inpu ids",input_ids)

        embedding_output = self.embeddings(input_ids)
        embedding_output=embedding_output.transpose(0,1)
        #print("emb out",embedding_output.size())


        if attention_mask is None:
            inputids=input_ids.transpose(1,0).squeeze(2)
            attention_mask=torch.ones_like(inputids)
            attention_mask[inputids==self.embeddings.word_padding_idx]=0
            #print("att before",attention_mask)
            #indices=input_ids==1
            #attention_mask[indices]=0
            #print("att after",attention_mask)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        #print("size of encoded layers, embeddings",encoded_layers.size(),embedding_output.size())
        return embedding_output.transpose(0,1),encoded_layers.transpose(0,1),lengths


class TransformerModel(nn.Module):
    def __init__(self, encoder:OnmtBertTransformerEncoder, decoder:onmt.decoders.TransformerDecoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        #print("lengths, src and tgt size is ",lengths.size(),src.size(),tgt.size())
        #src_enc=src.squeeze(2)

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state,memory_bank,lengths = self.encoder(input_ids=src,lengths=lengths)
        #print("size of encoded layers",memory_bank.size())

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        #print("decoded result",dec_out.size(),attns["std"].size())
        return dec_out, attns


class TextGeneratorModel(object):

    opt=None
    model_opt=None
    fields=None
    gpu_id=None
    device=None

    def __init__(self):
        self.build_model()

    def build_generator(self):
        model_opt=self.model_opt
        fields=self.fields

        from onmt.modules.util_class import Cast

        # Build Generator.
        if not model_opt.copy_attn:

            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size,
                          len(fields["tgt"].base_field.vocab)),
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_decoder_embeddings:
                generator[0].weight = self.transformer.decoder.embeddings.word_embeddings.weight
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = onmt.modules.CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

        return generator

    def use_device(self):
        from onmt.utils.misc import use_gpu
        gpu=use_gpu(self.opt)
        gpu_id=self.gpu_id
        if gpu and gpu_id is not None:
            device = torch.device("cuda", gpu_id)
        elif gpu and not gpu_id:
            device = torch.device("cuda")
        elif not gpu:
            device = torch.device("cpu")

        self.device=device

        self.transformer.to(device)
        if self.model_opt.model_dtype == 'fp16':
            self.transformer.half()

        logger.info("device:{},half:{}".format(device,self.model_opt.model_dtype))

    def build_model(self):

        #encoder
        bert=OnmtBertTransformerEncoder.from_pretrained(programmingalpha.BertBasePath)

        def __copyEmbeddings(embeddings:nn.Embedding,index1,index2):
            #print(embeddings.weight.size())
            weight=embeddings.weight.detach().numpy()

            weight[index2]=deepcopy(weight[index1])

            weight=torch.tensor(weight,requires_grad=True)
            embeddings.weight=nn.Parameter(weight,requires_grad=True)

        __copyEmbeddings(bert.embeddings.word_embeddings,0,1)
        __copyEmbeddings(bert.embeddings.word_embeddings,100,0)


        tgt_base_field = self.fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        model_dim= self.model_opt.rnn_size #if hasattr(self.model_opt,"rnn_size") else 768
        max_seq_length=5000
        max_relative_position=512
        heads=12
        feed_forwad_size=3072
        drop_rate=self.model_opt.dropout #if hasattr(self.model_opt,"dropout") else 0
        layers=self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        #tgt embeddings

        transformerEmb=OnmtBertEmbedding(vocab_size,model_dim,max_seq_length,drop_rate,pad_idx,bert.embeddings)

        bert.setEmbeddings(transformerEmb)
        #transformerEmb=BertAsEmbedding(bert,self.tgt_padding)


        #decoder
        transformerDecoder=onmt.decoders.TransformerDecoder(
            num_layers=layers, d_model=model_dim, heads=heads, d_ff=feed_forwad_size,
                         copy_attn=True, self_attn_type="scaled-dot", dropout=drop_rate, embeddings=transformerEmb,
                         max_relative_positions=max_relative_position
        )

        self.transformer=TransformerModel(bert,transformerDecoder)

        self.transformer.generator=self.build_generator()

        self.use_device()

    def loadModel(self,model_file=None,checkpoint=None):
        assert model_file is not None or checkpoint is not None

        if checkpoint is None:
            model_dict=torch.load(model_file)
        else:
            model_dict=checkpoint

        weight_dict=model_dict["model"]
        generator_dict=model_dict["generator"]
        #print(weight_dict.keys())
        #print(generator_dict.keys())
        for k in generator_dict:
            weight_dict["generator."+k]=generator_dict[k]

        transformer_dict=self.transformer.state_dict()
        exlude_dict=("encoder.embeddings.token_type_embeddings.weight","encoder.embeddings.position_embeddings.weight",
                     "decoder.embeddings.position_embeddings.weight")
        restore_dict={}

        layers= self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        logger.info("decoder layer num: %d"% layers)

        #self.transformer.load_state_dict(weight_dict)
        for k,v in weight_dict.items():
            if k in exlude_dict:
                logger.info("skip :{}".format(k))
                continue
            restore_dict[k]=v
        transformer_dict.update(restore_dict)


        if model_file:
            logger.info("init model weight with "+model_file)
        else:
            logger.info("init model with checkpoint")
