# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:08:55 2024

@author: saulo
"""

from pre_processador import PreProcessador
from metrificador import Metrificador

configs = {
    'col_classe': 'Class',
    'nome_arquivo': './Trabalhos/Trabalho 1/Classificação/airlines_delay.csv',
    'remover_colunas': ['Flight'],
    'concatenacao': None,#{
    #    'col1': 'AirportTo',
    #    'col2': 'AirportFrom',
    #    'concat': '-',
    #    'col_nova': 'Airports',
    #    'drop_cols': True
    #},
    'cols_categoria_nominal': ['AirportTo', 'AirportFrom'],
    'cols_dummy': ['Airline'],
    'padronizacao': True
}

processador = PreProcessador(configs)

# Configuração do KNN
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

acuracias_treinamento = []
acuracias_teste = []

# testar vários valores de K
numero_vizinhos = range(1,51)

for k in numero_vizinhos:
    # construir modelo
    classificador = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    #treinar modelo
    classificador.fit(processador.previsores_treinamento, processador.classe_treinamento)
    acuracias_treinamento.append(classificador.score(processador.previsores_treinamento, processador.classe_treinamento))
    acuracias_teste.append(classificador.score(processador.previsores_teste, processador.classe_teste))

plt.plot(numero_vizinhos, acuracias_treinamento, label='acuracia de treinamento')
plt.plot(numero_vizinhos, acuracias_teste, label='acuracia de teste')
plt.ylabel('Acuracia')
plt.xlabel('Valor de K')
plt.legend()

# Classificação com KNN 

n = 8

# minkowski com p=2 é a distância euclidiana
classificador = KNeighborsClassifier(n_neighbors=n, metric='minkowski', p=2)

# Treinamento

classificador.fit(processador.previsores_treinamento, processador.classe_treinamento)

# teste
previsoes = classificador.predict(processador.previsores_teste)

# análise de resultados
metrificador = Metrificador()

acuracia = metrificador.acuracia(previsoes, processador.classe_teste)
matriz = metrificador.matrizConfusao(previsoes, processador.classe_teste)
