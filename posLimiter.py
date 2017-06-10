# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 19:51:34 2017

@author: rushilcd
"""
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
def posLimiter(string):
    strng=''
    for itm in string:
        try:
            strng += str(itm)+' '
        except Exception:
            continue
    string=strng[:-1]
    limiters = ['NN', 'JJ', 'RB', 'VB']
    execfile('pos.py')
    try:
        l = pos(string)
        s = [word for word, p in zip(string.split(),l) if str(p).encode('UTF8') in limiters]
        
    except Exception:
        return '~exceptio~'
    return s




def pos(string):
    try:
        p = [x['pos'] for x in nlp.annotate(string, properties={
                       'annotators': 'tokenize,ssplit,pos,lemma',
                   'outputFormat': 'json'})['sentences'][0]['tokens']]
    except Exception:
        return None
    return p