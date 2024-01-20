# -*- UTF-8 -*-
from util.preprocessor import PreprocessPostContent


class Answer:

    def __init__(self, body, created_date, score=0, comment_count=0):
        self.body = body
        self.score = score
        self.comment_count = comment_count
        self.created_date = created_date

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        body = " ".join(body_para_list)
        return body