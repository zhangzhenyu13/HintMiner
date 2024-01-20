from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertEmbeddings,BertEncoder,BertPooler,BertConfig
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
import math

class BertForLinkRelationPrediction(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForLinkRelationPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits


#unlimited capacity model with performance drop
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
    def __init__(self, config:BertConfig,bertEmb:BertEmbeddings):
        super(OnmtBertEmbedding, self).__init__()
        #parameters
        self.max_position_size=config.max_position_embeddings*4
        self.vocab_size=config.vocab_size
        self.type_num=2
        self.dropout_prob=config.hidden_dropout_prob
        self.embeddings_size=config.hidden_size

        self.word_embeddings=bertEmb.word_embeddings
        self.position_embeddings=PositionalEncoding(dim=config.hidden_size,max_len=self.max_position_size)
        self.token_type_embeddings = bertEmb.token_type_embeddings

        self.LayerNorm = bertEmb.LayerNorm


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, input_ids, token_type_ids=None, step=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings  + token_type_embeddings

        embeddings=embeddings.transpose(0,1)
        #print("word emb",words_embeddings.size())

        embeddings = self.position_embeddings(embeddings,step)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print("emb",embeddings.size())
        #now is (len*b*dim)

        embeddings=embeddings.transpose(0,1)
        return embeddings


class AttnBertPooler(nn.Module):
    def __init__(self, config):
        super(AttnBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()
        self.hidden_size=config.hidden_size

    def forward(self, hidden_states,attention_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0].view(len(hidden_states),-1,1)
        #print("first token tensor",first_token_tensor.size())
        #print("mask",attention_mask.size())
        scores=torch.matmul(hidden_states[:,1:], first_token_tensor)/math.sqrt(self.hidden_size)
        #print("scores",scores.size())
        attn_token_tensor=torch.matmul( hidden_states[:,1:].view(hidden_states.size(0),self.hidden_size,-1), scores )
        #print("attention tensor1",attn_token_tensor.size())
        attn_token_tensor=attn_token_tensor.view( attn_token_tensor.size(0), self.hidden_size )
        #print("attention tensor2",attn_token_tensor.size())

        first_token_tensor=first_token_tensor.squeeze(2)
        pooled_token_tensor=torch.cat((attn_token_tensor,first_token_tensor),dim=-1)
        #attn_token_tensor=attn_token_tensor.unsqueeze(2)
        #pooled_token_tensor=torch.cat((attn_token_tensor,first_token_tensor),dim=1)

        #print("pooled tensor",pooled_token_tensor.size())
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class InferenceNet(BertPreTrainedModel):
    def __init__(self, config,num_labels):
        super(InferenceNet, self).__init__(config)
        embeddings = BertEmbeddings(config)
        self.embeddings=OnmtBertEmbedding(config,embeddings)
        self.attnpooler=AttnBertPooler(config)
        self.encoder = BertEncoder(config)
        self.pooler=BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
        self.num_labels=num_labels
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]
        pooled_output = self.attnpooler(sequence_output,attention_mask)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits
