# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:19:45 2024

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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# testar vários valores de profundidade (depth)

acuracias_treinamento = []
acuracias_teste = []

valor_profundidade = range(1,16)

for i in valor_profundidade:
    # construir modelo
    classificador = RandomForestClassifier(criterion='gini', max_depth=7, n_estimators=i, max_features=20, random_state=0)
    #treinar modelo
    classificador.fit(processador.previsores_treinamento, processador.classe_treinamento)
    acuracias_treinamento.append(classificador.score(processador.previsores_treinamento, processador.classe_treinamento))
    acuracias_teste.append(classificador.score(processador.previsores_teste, processador.classe_teste))

plt.plot(valor_profundidade, acuracias_treinamento, label='acuracia de treinamento')
plt.plot(valor_profundidade, acuracias_teste, label='acuracia de teste')
plt.ylabel('Acuracia')
plt.xlabel('Estimadores')
plt.legend()


# Geração da Random Forest
classificador = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=6, max_features=20, random_state=0)

# Treinamento

classificador.fit(processador.previsores_treinamento, processador.classe_treinamento)

# Teste
previsoes = classificador.predict(processador.previsores_teste)

# Análise de resultados
metrificador = Metrificador()

acuracia = metrificador.acuracia(previsoes, processador.classe_teste)
matriz = metrificador.matrizConfusao(previsoes, processador.classe_teste)