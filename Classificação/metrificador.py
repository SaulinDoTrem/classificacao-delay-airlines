# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:22:38 2024

@author: saulo
"""


from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac

class Metrificador:
        
    def matrizConfusao(self, previsoes, classes):
        return cm(classes, previsoes)
        
    def acuracia(self, previsoes, classes):
        return ac(classes, previsoes)