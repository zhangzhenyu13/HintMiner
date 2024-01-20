# -*- UTF-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from entity.answer import Answer
from entity.post import Post
from entity.question import Question
from retriver import search_es
from util import tokenizer
from util.preprocessor import PreprocessPostContent
import numpy as np


class Query:
    def __init__(self, title, body, tag_list, created_date):
        self.title = title
        self.body = body
        self.tag_list = tag_list
        self.created_date = created_date
        self.searched_post_list = []

    def search(self, size=200):
        search_result_list= search_es.search(self.title, size)
        # Post list
        post_obj_list = []
        for result in search_result_list:
            result = result['_source']
            # Question
            question = Question(result['question']['Title'], result['question']['Body'], result['question']['CommentCount'], result['question']['Score'], result['question']['Tags'], result['question']['CreationDate'])
            # Answer list
            # body, created_date, score=0, comment_count=0
            answers = result['answers']
            answer_list = []
            for answer in answers:
                # body, created_date, score=0, comment_count=0)
                answer = Answer(answer['Body'],answer['CreationDate'],answer['Score'],answer['CommentCount'])
                answer_list.append(answer)

            # Add Post to post list
            post_obj_list.append(Post(question, answer_list))

        self.searched_post_list = post_obj_list
        # 使用ES返回的 _score 值
        for i in range(len(search_result_list)):
            es_tfidf = search_result_list[i]['_score']
            self.searched_post_list[i].set_question_body_tfidf(es_tfidf)


    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        body = " ".join(body_para_list)
        return body

    def __calculate_a_title_relevance(self, question_obj):
        query_title_word_list = tokenizer.tokenize(self.title)
        if len(query_title_word_list) == 0:
            return 0

        question_title_word_list = tokenizer.tokenize(question_obj.title)
        # lower
        question_title_word_list = [w.lower() for w in question_title_word_list]
        query_title_word_list = [w.lower() for w in query_title_word_list]

        overlap = [value for value in query_title_word_list if value in question_title_word_list]
        ret = len(overlap) / len(query_title_word_list)

        return ret

    def calculate_title_relevance(self):
        for post in self.searched_post_list:
            post.set_title_relevance(self.__calculate_a_title_relevance(post.question_obj))

    def __calculate_a_tag_relevance(self, question_obj):
        if len(self.tag_list) == 0:
            return 0

        overlap = [value for value in self.tag_list if value in question_obj.tag_list]
        ret = len(overlap) / len(self.tag_list)

        return ret

    def calculate_tag_relevance(self):
        for post in self.searched_post_list:
            post.set_tag_relevance(self.__calculate_a_tag_relevance(post.question_obj))

    def calculate_tf_idf(self, type="question_body"):
        """
        计算TF_IDF值
        :param post_list:
        :param type: question_body or answer_body
        :return:
        """
        corpus = []
        # question body语料 每一行是每一个documents(首行是query body(处理后的))
        corpus.append(self.parse_body())
        if type == "question_body":
            for post in self.searched_post_list:
                corpus.append(post.question_obj.parse_body())
        elif type == "answer_body":
            for post in self.searched_post_list:
                corpus.append(post.concat_answer_body())
        else:
            raise NameError("Type Error")

        # print(corpus)
        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)  # 计算个词语出现的次数
        # print(vectorizer.get_feature_names()) # 获取词袋中所有文本关键词
        # print(X.toarray()) # 查看词频结果

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值
        tfidf_array = tfidf.toarray()
        row, col = tfidf_array.shape
        if type == "question_body":
            for i in range(1, row):
                sum = 0
                for j in range(col):
                    if tfidf_array[0, j] != 0:
                        sum += tfidf_array[i, j]

                self.searched_post_list[i - 1].set_question_body_tfidf(sum)
        else:
            for i in range(1, row):
                sum = 0
                for j in range(col):
                    if tfidf_array[0, j] != 0:
                        sum += tfidf_array[i, j]

                self.searched_post_list[i - 1].set_answer_body_tfidf(sum)

    def __calculate_a_score(self, post_obj, alpha):
        comment_count = post_obj.question_obj.comment_count
        vote_score = post_obj.question_obj.score
        for answer_obj in post_obj.answer_obj_list:
            comment_count += answer_obj.comment_count
            vote_score += answer_obj.score

        score = (1 - alpha) * comment_count + alpha * vote_score
        return score

    def calculate_score(self, alpha=0.8):
        for post in self.searched_post_list:
            post.set_score(self.__calculate_a_score(post, alpha))

    def __get_body_code(self):
        code_snippet_list = PreprocessPostContent().get_single_code(self.body)
        single_code_list = []
        for code_snippet in code_snippet_list:
            code_list = code_snippet.split()
            if len(code_list) == 1:
                single_code_list.extend(code_list)

        return single_code_list

    def __calculate_a_code_relevance(self, post_obj):
        query_code_list = self.__get_body_code()
        if len(query_code_list) == 0:
            return 0

        post_code_list = []
        post_code_list.extend(post_obj.get_question_body_code())
        post_code_list.extend(post_obj.get_answer_body_code())

        overlap = [code for code in query_code_list if code in post_code_list]
        code_relevance = len(overlap) / len(query_code_list)

        return code_relevance

    def calculate_code_relevance(self):
        for post in self.searched_post_list:
            code_relevance = self.__calculate_a_code_relevance(post)
            post.set_code_relevance(code_relevance)

    # Standard normalization
    def __normalize(self, score_list):
        return np.divide(np.subtract(score_list, np.average(score_list)), (np.std(score_list) + 0.0001))

    def normalized_post_score(self):
        title_relevance_list = []
        tag_relevance_list = []
        question_tfidf_list = []
        answer_tfidf_list = []
        score_list = []
        code_relevance_list = []

        for searched_post in self.searched_post_list:
            title_relevance_list.append(searched_post.title_relevance)
            tag_relevance_list.append(searched_post.tag_relevance)
            question_tfidf_list.append(searched_post.question_tfidf)
            answer_tfidf_list.append(searched_post.answer_tfidf)
            score_list.append(searched_post.score)
            code_relevance_list.append(searched_post.code_relevance)

        title_relevance_list = self.__normalize(title_relevance_list)
        tag_relevance_list = self.__normalize(tag_relevance_list)
        question_tfidf_list = self.__normalize(question_tfidf_list)
        answer_tfidf_list = self.__normalize(answer_tfidf_list)
        score_list = self.__normalize(score_list)
        code_relevance_list = self.__normalize(code_relevance_list)

        for i in range(len(self.searched_post_list)):
            self.searched_post_list[i].title_relevance = title_relevance_list[i]
            self.searched_post_list[i].tag_relevance = tag_relevance_list[i]
            self.searched_post_list[i].question_tfidf = question_tfidf_list[i]
            self.searched_post_list[i].answer_tfidf = answer_tfidf_list[i]
            self.searched_post_list[i].score = score_list[i]
            self.searched_post_list[i].code_relevance = code_relevance_list[i]

    def calculate_posts_all_score(self):
        for post in self.searched_post_list:
            post.cal_all_score()

    def range(self):
        self.calculate_title_relevance()
        # self.calculate_tf_idf(type="question_body")
        # self.calculate_tf_idf(type="answer_body")
        self.calculate_code_relevance()
        self.calculate_tag_relevance()
        self.calculate_score()
        self.normalized_post_score()
        self.calculate_posts_all_score()
        self.searched_post_list = sorted(self.searched_post_list, reverse=True)

if __name__ == '__main__':
    tag_list1 = ['Java', '<java>', 'println']
    tag_list2 = ['<c++>', '<java>', '<python>', 'pycharm']
    tag_list3 = ['<c++>', '<JAVA>', '<python>', 'pycharm']

    query = Query("How to use println in java", "Please show me how to use <code>println()<code> in java", tag_list1, "2019-5-16")
    query.search(size=2000)
    query.range()
    for pos in query.searched_post_list:
        print(pos.question_obj.title)
