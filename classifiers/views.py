from django.shortcuts import render
from .forms import ClassifyForm
from classifiers.GenreClassification.GenreClassifier import Genreclassifier

from classifiers.GhazalClassifiers.ParstBert import ParsBertClassifier
from classifiers.GhazalClassifiers.Albrt import AlbertClassifier
from classifiers.GhazalClassifiers.mBert import mBertClassifier
from classifiers.GhazalClassifiers.LSTM import LSTMClassifier
from classifiers.GhazalClassifiers.GRU import GRUClassifier
from classifiers.GhazalClassifiers.xlmRoberta import RobertaClassifier

from classifiers.RobaeeClassifiers.Albrt import RobaeeAlbertClassifier
from classifiers.RobaeeClassifiers.GRU import RobaeeGRUClassifier
from classifiers.RobaeeClassifiers.LSTM import RobaeeLSTMClassifier
from classifiers.RobaeeClassifiers.mBert import RobaeemBertClassifier
from classifiers.RobaeeClassifiers.ParstBert import RobaeeParsBertClassifier
from classifiers.RobaeeClassifiers.xlmRoberta import RobaeeRobertaClassifier

from classifiers.GhazalRobaeeClassifiers.parsBERT import PoemparsBertClassifier
from classifiers.GhazalRobaeeClassifiers.xlmRoBERTa import PoemrobertaClassifier

from django.contrib import messages

 
# Create your views here.
def classifyPoems(request):
    if request.method == 'POST':
        form = ClassifyForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            # print(data)
            result = Genreclassifier(data['text'])
            print(result)
            if result == "ghazal":
                if(data['model'] == "ParsBert"):
                    res = ParsBertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "Albert"):
                    res = AlbertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "mBert"):
                    res = mBertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "LSTM"):
                    res = LSTMClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "GRU"):
                    res = GRUClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "xlm-Roberta"):
                    res = RobertaClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                
            elif result == "robaee":
                if(data['model'] == "ParsBert"):
                    res = RobaeeParsBertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "Albert"):
                    res = RobaeeAlbertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "mBert"):
                    res = RobaeemBertClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "LSTM"):
                    res = RobaeeLSTMClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "GRU"):
                    res = RobaeeGRUClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                elif(data['model'] == "xlm-Roberta"):
                    res = RobaeeRobertaClassifier(data['text'])
                    predict = res[0]
                    prb = res[1]
                    print(predict)
                pass
        if result == 'ghazal':
            result = 'غزل'
        elif result == 'robaee':
            result = 'رباعی'

        if predict == 'attar':
            predict = 'عطار'
        elif predict == 'moulavi':
            predict = 'مولوی'
        elif predict == 'hafez':
            predict = 'حافظ'
        elif predict == 'saadi':
            predict = 'سعدی'
        elif predict == 'sanaee':
            predict = 'سنایی'
        elif predict == 'abusaeed':
            predict = 'ابوسعید'
        
        messages.add_message(request, messages.INFO, f"شعر در قالب  '{result}' سروده شده است.")
        messages.add_message(request, messages.INFO, f"   با احتمال {prb} شاعر این شعر '{predict}' است.")
        return render(request , 'index.html',{'form':form})
    else:
        form = ClassifyForm()
        return render(request , 'index.html',{'form':form})
    
