# -*- UTF-8 -*-
import os

import re
from textblob import TextBlob
from util import tokenizer


class PreprocessPostContent(object):
    code_snippet = re.compile(r"<pre>.*?</pre>")
    code_insider = re.compile(r"<code>.*?</code>")
    html_tag = re.compile(r"<.*?>")
    comment_bracket = re.compile(r"\(.*?\)")
    quotation = re.compile(r"(\'\')|(\")|(\`\`)")
    href_resource = re.compile(r"<a.*?>.*?</a>")
    paragraph = re.compile(r"<p>.*?</p>")
    equation1 = re.compile(r"\$.*?\$")
    equation2 = re.compile(r"\$\$.*?\$\$")
    integers = re.compile(r"^-?[1-9]\d*$")
    floats = re.compile(r"^-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$")
    operators = re.compile(r"[><\+\-\*/=]")
    email = re.compile(r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*")
    # web_url = re.compile(r"[a-zA-z]+://[^\s]*")
    web_url = re.compile(r"((http[s]{0,1}|ftp)://[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\.\-~!@#$%^&*+?:_/=<>]*)?)|((www.)|[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\.\-~!@#$%^&*+?:_/=<>]*)?)")
    punctuation = re.compile("[,!?:#$%^&*\']")

    @staticmethod
    def filterNILStr(s):
        filterFunc = lambda s: s and s.strip()
        s = ' '.join(filter(filterFunc, s.split())).strip()

        return s

    def __init__(self):
        # self.max_quote_rate = 1.5
        # self.max_quote_diff = 5
        # self.min_words_sent = 5
        # self.min_words_paragraph = 10
        self.max_number_pragraph = 5
        self.num_token = "[NUM]"
        self.code_token = "[CODE]"
        # self.max_code = 5

    def remove_bracket(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s = sent.string
            s_c = re.sub(self.comment_bracket, "", s)
            s_c_words = TextBlob(s_c).words

            cleaned.append(s_c)

        return " ".join(cleaned)

    def remove_quotation(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s_c = re.sub(self.quotation, "", sent.string)
            cleaned.append(s_c)

        return " ".join(cleaned)

    def remove_href(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s = sent.string
            s_c = re.sub(self.href_resource, "", s)
            if sent.words != TextBlob(s_c).words:
                continue
            cleaned.append(s)

        return " ".join(cleaned)

    def remove_code(self, txt):
        cleaned = re.sub(self.code_snippet, "", txt)
        cleaned = re.sub(self.code_insider, "", cleaned)
        return cleaned

    def remove_equation(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s = sent.string
            s_c = re.sub(self.equation2, "", s)
            s_c = re.sub(self.equation1, "", s_c)
            if sent.words != TextBlob(s_c).words:
                continue
            cleaned.append(s)

        return " ".join(cleaned)

    def remove_numbers(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:

            s_tokens = sent.string.split()
            # print("in num",s_tokens)
            s_tokens = list(map(lambda t: re.sub(self.floats, "", t), s_tokens))
            # print("float",(s_tokens))
            s_tokens = map(lambda t: re.sub(self.integers, "", t), s_tokens)
            s_c = " ".join(s_tokens)
            # print("in number=>",sent.string,"--vs--",s_c)

            if len(sent.words) - len(TextBlob(s_c).words) > self.max_number_pragraph:
                continue

            s_tokens = sent.string.split()
            s_tokens = map(lambda t: re.sub(self.floats, " %s " % self.num_token, t), s_tokens)
            s_tokens = map(lambda t: re.sub(self.integers, " %s " % self.num_token, t), s_tokens)
            s_c = " ".join(s_tokens)

            cleaned.append(s_c)

        cleaned = " ".join(cleaned)

        return cleaned

    def remove_operators(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s = sent.string
            s_c = re.findall(self.operators, s)
            if len(s_c) > 3:
                continue
            cleaned.append(s)

        return " ".join(cleaned)

    def remove_hmtltag(self, txt):
        cleaned = re.sub(self.html_tag, "", txt)
        return cleaned

    def remove_email(self, txt):
        cleaned = []
        for sent in TextBlob(txt).sentences:
            s = sent.string
            s_c = re.sub(self.email, "", s)
            if sent.words != TextBlob(s_c).words:
                continue
            cleaned.append(s)
        return " ".join(cleaned)

    def remove_url(self, txt):
        cleaned = re.sub(self.web_url, "", txt)
        return cleaned

    def remove_punctuation(self, txt):
        cleaned = re.sub(self.punctuation, "", txt)
        return cleaned

    def __process(self, txt):
        # print("\nprocess=>",txt)

        # txt = self.remove_emb_code(txt)
        # print("after rm codecs=>",txt)

        txt = self.remove_href(txt)
        # print("after rm href=>",txt)

        txt = self.remove_email(txt)
        # print("atfer rm email=>",txt)

        txt = self.remove_url(txt)
        # print("atfer rm url=>",txt)

        txt = self.remove_hmtltag(txt)
        # print("after rm tag=>",txt)

        # txt = self.remove_equation(txt)
        # print("atfer rm eq=>",txt)

        txt = self.remove_bracket(txt)
        # print("after rm quote=>",txt)

        txt = self.remove_numbers(txt)
        # print("after rm num=>",txt)

        txt = self.remove_operators(txt)
        # print("after rm ops=>",txt)

        txt = self.remove_quotation(txt)
        # print("atfer rm quotation=>",txt)

        # txt = self.remove_useless(txt)
        # print("after rm use=>",txt)

        txt = self.remove_punctuation(txt)

        return txt


    def getParagraphs(self, raw_txt):
        """
        :param raw_txt:
        :return: a paragraph list
        """
        raw_txt = self.filterNILStr(raw_txt)
        paragraphs_candiates = re.findall(self.paragraph, raw_txt)
        paragraphs_candiates = [p[3:-4] for p in paragraphs_candiates if len(p[3:-4]) > 0]
        paragraphs = []
        for p in paragraphs_candiates:
            paragraphs.append(p)

        return paragraphs

    def getProcessedParagraphs(self, raw_txt):
        paragraphs = self.getParagraphs(raw_txt)
        processed_paragraphs = []
        for p in paragraphs:
            p1 = self.remove_code(p)
            processed_paragraphs.append(self.__process(p1))

        return processed_paragraphs

    # 过滤列表中单字符（标点）
    def filter_wordlist(self, wordlist):
        condition = lambda t: len(t) > 1 or t == 'I'
        filter_list = list(filter(condition, wordlist))
        return filter_list

    # [ [word1, word2, ...], [word1, word2, ...], [word1, word2, ...], ... ]
    def get_mul_para_wordlist_list(self, raw_txt):
        # return a list of paragraphs of plain text(word list)
        txt = self.remove_code(raw_txt)

        paragraphs = self.getParagraphs(txt)

        wordlist_list = []
        for p in paragraphs:
            cleaned = self.__process(p)
            wordlist = self.filterNILStr(cleaned)
            wordlist = tokenizer.tokenize(wordlist)
            wordlist = self.filter_wordlist(wordlist)
            wordlist_list.append(wordlist)

        return wordlist_list

    # [word1, word2, ...]
    def get_single_para_word_list(self, raw_txt):
        # return a list of plain text(word list)
        # filter code
        txt = self.remove_code(raw_txt)

        cleaned = self.__process(txt)
        text = self.filterNILStr(cleaned)
        word_list = tokenizer.tokenize(text)
        word_list = self.filter_wordlist(word_list)
        return word_list

    def get_single_code(self, raw_txt):
        ret = re.findall(re.compile(r"<code>(.*?)</code>"), raw_txt)
        return ret

if __name__ == '__main__':
    ans = '''
    <p>In linear regression, the word "linear" applies to the coefficients: the dependence between $Y$ and the coefficients is linear. This does not mean the dependence between $Y$ and $X$ is linear.</p>
    <p>I think a linear regression is handling the regression problem using math:</p>
    <p>Assume $X$ is a one dimensional variable. Basic linear regression is (I omit the noise and intercept for simplicity):
    $$Y=\beta X$$</p>
    <p> as a result of this, you better visit www.baidu.com to get full answer</p>
        <p> as a result of this, you better visit https://www.baidu.com to get full answer</p>

    <p>But this is still linear regression:</p>
    
    <p>$$Y=\beta_1 X+\beta_2X^2+\beta_3\log(X)$$</p>
    
    <p>The latter is the same as basic linear regression with feature vector $(X,X^2,\log(X))$ instead of $X$.</p>
    
    <p>By linear regression I assume that you mean simple linear regression. The difference is in the number of independent explanatory variables you use to model your dependent variable.</p>

    <p>Simple linear regression</p>
    
    <p>$Y=\beta X+\beta_0$</p>    
    
    <p>I have been given 65 values. 57 of these data values are quarterly results and 8 are the holdback data to be used. </p>
    
    <p>I have to do: 
    - Regression with Dummy variables with a linear trend cycle component </p>
    
    <p>Does anyone know what to do as my results aren't making much sense? </p>
    
    <p>For the first part - I obviously split the data into dummy variables for the relevant quarters (Q1-Q4). </p>
    
    <p>I then performed regression analysis - linear. But all my values are extremely large and not significant. Also Q2 has been listed as 'excluded variables' in the results? I have followed the steps and I am unsure why this has happened. </p>
    
    <p>Then I thought of removing Q4, due to multi-collinearity but again the values are still quite large (>.450). </p>
    
    <p>Not sure if I am doing something wrong at the start (especially with the excluded variables aspect) </p>
    
    <p>Anybody got any idea? This is driving me nuts</p>
    
    <p>Update: It won't let me comment back on the main page for some reason. </p>
    
    <p>The data set was given to us:
    "It is a quarterly series of total consumer lending. It is not seasonally adjusted.
    The first 57 data values for modelling and choose the remaining 8 data values as holdback data to test your models."</p>
    
    <p>The data is: 
    16180
    17425
    43321
    3214 4324 5435 41 143221 545 45</p>
    
    <p>It has to be SPSS generated: as it is not like.</p>
    
    <p>Email primarybeing12@hotmail.co.uk - not letting me respond to people. Thanks for any help!</p>
    
    <p>Doing the ARIMA forecasting is the next step (which I understand). I have to do regression on the linear/non-linear for this question</p>
    
    <p>If I was to use time, time^2, Q1, Q2, Q3 + lagged variables.</p>
    
    <p>Would I use lagged variables 1-3? Also, I understand the rest, but what benefit does using lagged variables do? As I said, feel free to e-mail me if you can.</p>
    
    <p>(I'm not positive about this, but...)</p>

    <p>AS3 uses a non-deterministic garbage collection. Which means that unreferenced memory will be freed up whenever the runtime feels like it (typically not unless there's a reason to run, since it's an expensive operation to execute). This is the same approach used by most modern garbage collected languages (like C# and Java as well).</p>
    
    <p>Assuming there are no other references to the memory pointed to by <code>byteArray</code> or the items within the array itself, the memory will be freed at some point after you exit the scope where <code>byteArray</code> is declared.</p>
    
    <p>You can force a garbage collection, though you really shouldn't. If you do, do it only for testing... if you do it in production, you'll hurt performance much more than help it.</p>
    
    <p>To force a GC, try (yes, twice):</p>
    
    <pre><code>flash.system.System.gc();
    flash.system.System.gc();
    </code></pre>
    
    <p><a href="http://www.craftymind.com/2008/04/09/kick-starting-the-garbage-collector-in-actionscript-3-with-air/" rel="noreferrer">You can read more here</a>.</p>

    '''

    # s='''<p><code>(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])</code></p>'''
    # ans+=s

    ans2 = '''
        <p>It is saied that Whenever<code>code</code> a problem becomes solvable by a computer, people start arguing that it does not require intelligence . </p>
        <p>[CLS] "Whenever a problem becomes solvable by a computer , people start arguing that it does not require intelligence . [SEP] John McCarthy is often quoted : `` As soon as it works , no one calls it AI anymore '' ( Referenced in CACM )[SEP] ."</p> 
        
        <p>"One of my teachers in <code>jet.listen</code>college said that in the 1950 's , a professor was asked what he thought was intelligent for a machine . The professor reputedly answered that if a vending machine gave him the right change , that would be intelligent ."</p> 
        
        <p>"Later , playing chess was considered intelligent . However , computers can now defeat grandmasters at chess , and people are no longer saying that it is a form of intelligence ."</p> 
        
        <p>"Now we have OCR . It 's already stated in another answer that our methods do not have the recognition facilities of a 5 year old . As soon as this is achieved , people will say `` meh , that 's not intelligence , a 5 year old can do that ! ''"</p> 
        
        <p>"A psychological bias , a need to state that we are somehow superior to machines , is at the basis of this ."</p>
    '''
    gold = '''
        <p>An optimal solution for the task as stated , would be some alignment algorithm like Smith-Waterman , with a matrix which encodes typical typo frequencies .</p>
        <p>As an exercise in NNs , I would recommend using a RNN . This circumvents the problem that your inputs will be of variable size , because you just feed one letter after another and get an output once you feed the delimiter .</p> 
        <p>As trainingsdata you 'll need a list of random words and possibly a list of random strings , as negative examples and a list of slightly messed up versions of your target word as positive examples .</p>
        <p>Here is a minimal character-level RNN , which consists of only a little more than a hundred lines of code , so you might be able to get your head around it or at least get it to run . Here is the excellent blog post by Karpathy to which the code sample belongs .</p>
    '''

    text1 = "<p>Who was Jim Henson</p><p>Jim Henson was a puppeteer.But he always using pytorch and java's code: <code>System.out.println()</code></p>"

    text2 = "I am a bot ,and he's my friend, i. . # AI Beginners Book"

    text3 = "This is my<code>code1 's content</code>,and this is my<code>code2</code>,<code>Syste.println()</code>"

    # ans = PreprocessPostContent().get_mul_para_wordlist_list(ans)
    # ans = PreprocessPostContent().get_single_para_word_list(text)

    # ans = PreprocessPostContent().getProcessedParagraphs(text1)

    # ans = PreprocessPostContent().get_single_para_word_list(ans[0])

    ans = PreprocessPostContent().get_single_code(text3)

    for a in ans:
        print(a.split())
    # print(PreprocessPostContent().test())
