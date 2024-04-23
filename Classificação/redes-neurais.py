# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:44:20 2024

@author: saulo
"""

caminhoArquivo = './Trabalho 1/Classificação'

configs = {
    'col_classe': 'Class',
    'nome_arquivo': caminhoArquivo+'/airlines_delay.csv',
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

############## Classificação com Redes Neurais ##########

classificador = Classificador(processador)

classificador.RedesNeurais(verbose = True,
                              max_iter= 2500,
                              tol = 0.0000001,
                              solver = 'sgd',
                              hls=[10,10],
                              activation='relu',
                              random_state = 1)


#análise de resultados
metrificador = Metrificador()
acuracia = metrificador.acuracia(processador.classe_teste,classificador.previsoes)
matriz = metrificador.matrizConfusao(processador.classe_teste,classificador.previsoes)