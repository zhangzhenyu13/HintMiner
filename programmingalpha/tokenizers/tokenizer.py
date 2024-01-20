import json
import pexpect
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Tokenizer(object):
    def tokenize(self):
        raise NotImplementedError
    def convert_tokens_to_ids(self,tokens):
        raise NotImplementedError
    def convert_ids_to_tokens(self,ids):
        raise NotImplementedError

class CoreNLPTokenizer(object):

    def __init__(self):
        """
        Args:
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        #for file in `find /home/zhangzy/stanford-corenlp-full-2018-10-05/ -name "*.jar"`; do export
        #CLASSPATH="$CLASSPATH:`realpath $file`"; done
        import os
        path="/home/LAB/zhangzy/stanford-corenlp-full-2018-10-05/"
        files=os.listdir(path)
        jars=[]
        for f in files:
            if f[-4:]==".jar":
                jars.append(os.path.join(path,f))

        self.classpath = ":".join(jars)
        self.mem = "4g"
        self._launch()
        logger.info("init core_nlp tokenizer finished!")

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']

        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            return [token]

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{\r\n  "sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        tokens = tuple([self._convert(t["word"]) for s in output['sentences'] for t in s['tokens']])
        #tokens = tuple([t["word"] for s in output['sentences'] for t in s['tokens']])

        return tokens


import spacy

class SpacyTokenizer(object):

    def __init__(self):
        """
        Args:
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = 'en'
        nlp_kwargs = {'parser': False}
        nlp_kwargs['tagger'] = False
        nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)
        logger.info("init spacy tokenizer finished!")

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = tuple(map(lambda t:t.text,self.nlp.tokenizer(clean_text)))

        return tokens


class SimpleTokenizer(object):
    def tokenize(self,txt):
        return txt.split()


if __name__ == '__main__':
    tokenizer1=CoreNLPTokenizer()
    tokenizer2=SpacyTokenizer()
    s="I am a very powerful! (greatest) man"
    print(tokenizer1.tokenize(s))
    print(tokenizer2.tokenize(s))
    from programmingalpha.tokenizers import ngrams
    for n in range(1,5):
        print(ngrams(tokenizer2.tokenize(s),n=n))
