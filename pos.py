# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:56:44 2017

@author: rushilcd
"""

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
def pos(string):
    try:
        p = [x['pos'] for x in nlp.annotate(string, properties={
                       'annotators': 'tokenize,ssplit,pos,lemma',
                   'outputFormat': 'json'})['sentences'][0]['tokens']]
    except Exception:
        return None
    return p


