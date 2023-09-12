from django import forms

class ClassifyForm(forms.Form):
    modelsName = (
        ('ParsBert','ParsBert'),
        ('Albert','Albert'),
        ('mBert', 'mBert'),
        ('LSTM','LSTM'),
        ('GRU','GRU'),
        ('xlm-Roberta','xlm-Roberta'),
    )
    # fields = ('text', 'model')
    # widgets = {
    #     'text' : forms.Textarea(attrs={'class': 'form-control'}),
    #     'model': forms.Select(attrs={'class': 'form-control'})
    # }
    text = forms.CharField(widget=forms.Textarea, label="شعر")
    model = forms.ChoiceField( choices = modelsName, label="مدل انتخابی")