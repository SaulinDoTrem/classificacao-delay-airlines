# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:29:23 2024

@author: thais
"""

configs = {
    'col_classe': 'Class',
    'nome_arquivo': 'D:/dev/projetos/machineLearning/classificacao-delay-airlines/Classificação/airlines_delay.csv',
    'remover_colunas': [],
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


# Configuração da Árvore de Decisão
from sklearn.svm import SVC

# Geração SVM
classificador = SVC(kernel='rbf', gamma = 'auto', C = 0.1, random_state=1)

# Treinamento

classificador.fit(processador.previsores_treinamento, processador.classe_treinamento)

# Teste
previsoes = classificador.predict(processador.previsores_teste)

# Análise de resultados
metrificador = Metrificador()

acuracia = metrificador.acuracia(previsoes, processador.classe_teste)
matriz = metrificador.matrizConfusao(previsoes, processador.classe_teste)
