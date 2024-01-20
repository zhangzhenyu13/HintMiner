# -*- UTF-8 -*-

from util.preprocessor import PreprocessPostContent


class Question:

    def __init__(self, title, body, comment_count, score, tag_list, created_date):
        self.title = title
        self.body = body
        self.comment_count = comment_count
        self.score = score
        self.tag_list = tag_list
        self.created_date = created_date

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        body = " ".join(body_para_list)
        return body


