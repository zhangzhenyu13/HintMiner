from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from functools import partial
from programmingalpha.tokenizers.tokenizer import SimpleTokenizer
from nltk.translate.bleu_score import corpus_bleu
class LanguageMetrics(object):
    bleu = BLEUCalculator(tokenizer=SimpleTokenizer())
    rouge = RougeCalculator(stopwords=True, lang="en",tokenizer=SimpleTokenizer())

    @staticmethod
    def _computeScore(summary,refs,criteria):
        if isinstance(refs,str):
            refs=[refs]
        score=criteria(summary=summary,references=refs)
        return score

    @staticmethod
    def blue_score(summary,refs):
        score = LanguageMetrics._computeScore(summary,refs,LanguageMetrics.bleu.bleu)
        return score

    @staticmethod
    def rouge_1_score(summary,refs):
        score=LanguageMetrics._computeScore(summary,refs,LanguageMetrics.rouge.rouge_1)
        return score

    @staticmethod
    def rouge_2_score(summary,refs):
        score=LanguageMetrics._computeScore(summary,refs,LanguageMetrics.rouge.rouge_2)
        return score

    @staticmethod
    def rouge_l_score(summary,refs):
        score=LanguageMetrics._computeScore(summary,refs,LanguageMetrics.rouge.rouge_l)
        return score

    @staticmethod
    def rouge_be_score(summary,refs):
        score=LanguageMetrics._computeScore(summary,refs,LanguageMetrics.rouge.rouge_be)
        return score

    @staticmethod
    def rouge_n_score(summary,refs,n):
        rouge_n=partial(func=LanguageMetrics.rouge.rouge_n,n=n)
        score=LanguageMetrics._computeScore(summary,refs,rouge_n)
        return score







