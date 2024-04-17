# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:16:35 2024

@author: saulo
"""

from pre_processador import PreProcessador
from metrificador import Metrificador
import pandas as pd

configs = {
    'col_classe': 'Class',
    'nome_arquivo': './Trabalhos/Trabalho 1/Classificação/airlines_delay.csv',
    'remover_colunas': [],
    'concatenacao': None,#{
    #    'col1': 'AirportTo',
    #    'col2': 'AirportFrom',
    #    'concat': '-',
    #    'col_nova': 'Airports',
    #    'drop_cols': True
    #},
    'cols_categoria_nominal': [],
    'cols_dummy': [],
    'padronizacao': False
}

processador = PreProcessador(configs)

# Classificação por classe majoritária

# resultado mínimo
contagem = processador.classe_treinamento[processador.cols_classe[0]].value_counts()
classe_majoritaria = contagem.idxmax()

previsoes = pd.Series([classe_majoritaria]).repeat(processador.classe_teste.size)

# análise de resultados
metrificador = Metrificador()

matriz_confusao = metrificador.matrizConfusao(previsoes, processador.classe_teste)
# 74936	0
# 59910	0

acuracia = metrificador.acuracia(previsoes, processador.classe_teste) # 0.5557154086884297
