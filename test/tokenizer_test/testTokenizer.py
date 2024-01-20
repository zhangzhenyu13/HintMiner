from programmingalpha.tokenizers import (
    BertTokenizer,SpacyTokenizer,CoreNLPTokenizer, TransfoXLTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer)

import programmingalpha

def testBasicFunctions():
    #s="Today I faced the similar problem . I want to share my research , maybe it will be helpful for somebody . The symptoms : When you use any java software you get pop-up `` Applet alert '' `` Applet is attempting to ... '' BUT you know that you do n't use any applets . If you press the `` Stop Applet '' button on pop-up or suppress pop-up using or other way you will see in the log an error like or You can see that pop-up is created by the code from package . The reason : You are using jar that was downloaded from internet through Trend Micro InterScan Web Security Suite/Appliance proxy . This proxy make hooks in all jars that you download from internet , so that when they try to access files you see pop-up `` Applet alert '' . You may determine the affected jar by last stack-trace entrance before package . ( In my case it is from commons-digester.jar , in masch 's case it is from spring.jar ) Solution : You have two options : Problem description on Trend Micro web site"
    #s="0000 0000 0000 0000"
    #s="why is scan ##f ( ) causing infinite loop in this code ? why is scan # # f causing infinite loop in this code ?"
    s="assuming that he does not want to lose his driver \u2019 s licence"
    print('\n test bert tokenizer')
    tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath)
    print(tokenizer.tokenize(s))

    print('\n test spacy tokenizer')
    tokenizer=SpacyTokenizer()
    print(tokenizer.tokenize(s))

    print('\n test corenlp tokenizer')
    tokenizer=CoreNLPTokenizer()
    print(tokenizer.tokenize(s))

    print('\n test transformerXL tokenizer')
    tokenizer=TransfoXLTokenizer()
    print(tokenizer.tokenize(s))

    print('\n test openai tokenizer')
    tokenizer=OpenAIGPTTokenizer.from_pretrained(programmingalpha.openAIGPTPath)
    print(tokenizer.tokenize(s))

    print('\n test gpt2 tokenizer')
    tokenizer=GPT2Tokenizer.from_pretrained(programmingalpha.GPT2Path)
    print(tokenizer.encode(s))

def testConfig():
    tokenizer=BertTokenizer(programmingalpha.GloveStack+"vocab.txt")
    print(tokenizer.vocab)
if __name__ == '__main__':
    testBasicFunctions()
    #testConfig()
