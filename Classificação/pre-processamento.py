# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:49:32 2024

@author: Aluno
"""

from pre_processador import PreProcessador

# ['Flight', 'Time', 'Length', 'AirportFrom', 'AirportTo', 'Airline', 'DayOfWeek']

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
    'cols_categoria_nominal': ['AirportFrom', 'AirportTo', 'Airline'],
    'cols_dummy': [],
    'padronizacao': True
}

processador = PreProcessador(configs)
