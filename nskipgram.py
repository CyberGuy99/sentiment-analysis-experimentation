# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:11:48 2017

@author: rushilcd
"""


#original/restrictive skipgram
#def nskipgram(k,n,words_list):
#    group = []
#    for i in range(len(words_list)-n+1):
#        s = []
#        s.append(words_list[i])
#        for j in range(i+k+1,len(words_list),k+1):
#            if(j<len(words_list)):
#                s.append(words_list[j])
#        if(len(s)>=n):
#            group.append('-'.join(s[0:n]))       
#    return group

#inclusive/non-consecutive skipgram
def nskipgram(k,n,words_list):
    group = []
    for i in range(len(words_list)-n+1):
        s = []
        s.append(words_list[i])
        for j in range(i+k+1,len(words_list),k+1):
            if(j<len(words_list)):
                s.append(words_list[j])
        if(len(s)>=n):
            group.append('-'.join(s[0:n]))       
    if k>0:
        return group + nskipgram(k-1,n,words_list)
    else:
        return group


