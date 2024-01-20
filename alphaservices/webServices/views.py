from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from .qamodel import process
from html import escape
#def process(q):return "No"
# Create your views here.

def index(request:HttpRequest):
    #msg="hello, welcome to programming alpha for AI engineers and learners !!!"
    #return HttpResponse(msg)
    return render(request,"index.html")

def getAnswer(request:HttpRequest):
    request.encoding='utf-8'
    ans=None
    if 'q' in request.POST:
        question=request.POST['q']
        message = 'the quetsion you asked is ' + question
        if question and question.strip()!='':
            ans=process(question)
        else:
            ans='cannot answer blank questions!!!'
        #ans=escape(ans)
    else:
        message = 'cannot answer blank questions!!!'

    reply ={}
    if request.POST:
        reply['answer'] = ans if ans else message

    #print("request body=>",request.body)
    #print("ans is")
    #print(ans)
    return render(request, "alpha-QA.html", reply)
