# -*- coding: utf-8 -*-

caminhoArquivo = './Trabalhos/Trabalho 1/Classificação/'
arquivo = 'airlines_delay.csv'

configs = {
    'col_classe': 'Class',
    'nome_arquivo': caminhoArquivo+arquivo,
    'remover_colunas': ['Flight'],
    'concatenacao': {
        'col1': 'AirportTo',
        'col2': 'AirportFrom',
        'concat': '-',
        'col_nova': 'Airports',
        'drop_cols': True
    },
    'cols_categoria_nominal': ['Airports'],
    'cols_dummy': ['Airline'],
    'padronizacao': False
}

processador = PreProcessador(configs)


# Configuração do Naive Bayes
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


classificador = GaussianNB()

# Treinamento
classificador.fit(processador.previsores_treinamento, processador.classe_treinamento.iloc[:, 0])


# teste
previsoes = classificador.predict(processador.previsores_teste)

# análise de resultados
metrificador = Metrificador()

acuracia = metrificador.acuracia(previsoes, processador.classe_teste)
matriz = metrificador.matrizConfusao(previsoes, processador.classe_teste)
