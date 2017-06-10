# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 21:56:37 2017

@author: rushilcd
"""
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

def lemmatize(string):
    try:
        l = [x['lemma'] for x in nlp.annotate(string, properties={
                       'annotators': 'tokenize,ssplit,pos,lemma',
                   'outputFormat': 'json'})['sentences'][0]['tokens']]
    except Exception:
        return None
    return l