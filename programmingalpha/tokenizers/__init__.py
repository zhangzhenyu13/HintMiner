from .tokenizer import CoreNLPTokenizer,SpacyTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.tokenization_openai import OpenAIGPTTokenizer
from pytorch_pretrained_bert.tokenization_transfo_xl import TransfoXLTokenizer

def ngrams(words, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        if uncased:
            words =tuple(map(lambda s:s.lower(),words))

        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams


def get_class(name):
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'spacy':
        return SpacyTokenizer
    if name=='bert':
        return BertTokenizer
    if name=='gpt2':
        return GPT2Tokenizer
    if name=='openai':
        return OpenAIGPTTokenizer
    if name=='transformerXL':
        return TransfoXLTokenizer

    raise RuntimeError('Invalid retriever class: %s' % name)
