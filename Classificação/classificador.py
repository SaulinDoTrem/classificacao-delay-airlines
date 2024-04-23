# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:50:59 2024

@author: saulo
"""

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
class Classificador:
    
    def __init__(self, processador):
        self.cols_previsores = processador.cols_previsores
        self.previsores_treinamento = processador.previsores_treinamento
        self.previsores_teste = processador.previsores_teste
        self.classe_treinamento = processador.classe_treinamento
        self.classe_teste = processador.classe_teste
    
    def redesNeurais(self, verbose, max_iter, tol, solver, hls, activation, random_state):
        self.classificador = MLPClassifier(verbose = verbose,
                              max_iter= max_iter,
                              tol = tol,
                              solver = solver,
                              hidden_layer_sizes=hls,
                              activation=activation,
                              random_state = random_state)
        
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento)
        self.previsoes = self.preverModelo(self.previsores_teste)
        
        plt.imshow(self.classificador.coefs_[0], interpolation='none', cmap='viridis')
        plt.yticks(range(len(self.cols_previsores)), self.cols_previsores)
        plt.xlabel("Neuronios da primeira camada")
        plt.ylabel("Característica")
        plt.colorbar()
        
    def KNN(self, n):
        # minkowski com p=2 é a distância euclidiana
        self.classificador = KNeighborsClassifier(n_neighbors=n, metric='minkowski', p=2)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento)
        
        # teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
    def NaiveBayes(self):
        self.classificador = GaussianNB()
        
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])


        # teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
    def treinarModelo(self, previsores, classe):
        self.classificador.fit(previsores, classe)
        
    def preverModelo(self, previsores):
        self.classificador.predict(previsores)