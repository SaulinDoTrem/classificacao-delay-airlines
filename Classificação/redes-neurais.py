# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:44:20 2024

@author: saulo
"""

import sys
sys.path.append('D:/dev/projetos/machineLearning/classificacao-delay-airlines/Classificação')

from pre_processador import PreProcessador
from classificador import Classificador
from validacao_cruzada import ValidacaoCruzada
from metrificador import Metrificador


### PROCESSADOR ###
caminhoArquivo = 'D:/dev/projetos/machineLearning/classificacao-delay-airlines'

configs = {
    'col_classe': 'Class',
    'nome_arquivo': caminhoArquivo+'/Classificação/airlines_delay.csv',
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


### CLASSIFICAÇÃO ###
classificador = Classificador(processador)
classificador.RedesNeurais(verbose = True,
                              max_iter= 2500,
                              tol = 0.0000001,
                              solver = 'sgd',
                              hls=[10,10],
                              activation='relu',
                              random_state = 1)


### ANÁLISE DE RESULTADOS ###
metrificador = Metrificador()
acuracia = metrificador.acuracia(processador.classe_teste, classificador.previsoes)
matriz = metrificador.matrizConfusao(processador.classe_teste, classificador.previsoes)


### VALIDAÇÃO CRUZADA ###
validacaoCruzada = ValidacaoCruzada(classificador, processador)

validacaoCruzadaRedesNeurais = {
    'matriz_media': validacaoCruzada.matriz_media,
    'matriz_desvio_padrao': validacaoCruzada.matriz_desvio_padrao,
    'acuracia_final_media': validacaoCruzada.acuracia_final_media,
    'acuracia_final_desvio_padrao': validacaoCruzada.acuracia_final_desvio_padrao,
    'metricas_medias': validacaoCruzada.metricas_medias,
    'metricas_desvio_padrao': validacaoCruzada.metricas_desvio_padrao
    }
