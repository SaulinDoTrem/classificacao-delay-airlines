# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:12:32 2024

@author: Aluno
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

class PreProcessador:
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    label_binarizer = LabelBinarizer()
    
    
    def __init__(self, configs):
        self.prepararBase(configs)
        self.separarPrevisoresDaClasse(configs['col_classe'])
        self.preProcessar(configs)
        
         
    def prepararBase(self, configs):
        self.lerCsv(configs['nome_arquivo'])
        self.resumo = self.resumoBase()
        
        for col in configs['remover_colunas']:
            self.removerColuna(col)
            
        if configs['concatenacao'] != None:
            c = configs['concatenacao']
            self.concatenarColunas(c['col1'], c['col2'], c['concat'], c['col_nova'], c['drop_cols'])

    def lerCsv(self, nome_arquivo):
        self.base = pd.read_csv(nome_arquivo)
    
    def resumoBase(self):
        return self.base.describe()
    
    def removerColuna(self, nome_coluna):
        self.base.drop(nome_coluna, axis=1, inplace=True)
    
    def concatenarColunas(self, col1, col2, concatenador, nome_col_nova, remover_cols):
        concatenacao = self.base[col1]+concatenador+self.base[col2]
        self.base.insert(2, nome_col_nova, concatenacao, True)
        
        if remover_cols:
            self.removerColuna(col1)
            self.removerColuna(col2)
            
    def separarPrevisoresDaClasse(self, coluna_classe):
        colunas = self.base.columns
        
        self.cols_previsores = colunas.drop(coluna_classe).tolist()
        
        self.cols_classe = [coluna_classe]
        
        self.previsores = self.base[self.cols_previsores]
        self.classe = self.base[self.cols_classe]
        
    def transformarVariavelCategoriaNominal(self, nome_col):
        self.previsores.loc[:, nome_col] = self.label_encoder.fit_transform(self.previsores.loc[:, nome_col])
        
    def transformarVariavelDummy(self, nome_col):
        variaveis_dummy = self.label_binarizer.fit_transform(self.previsores[nome_col])
        novas_variaveis_dummy = self.label_binarizer.classes_
        df_variaveis_dummy = pd.DataFrame(variaveis_dummy, columns=novas_variaveis_dummy)
        self.previsores = self.previsores.join(df_variaveis_dummy)
        self.previsores.drop(nome_col, axis=1, inplace=True)
        self.cols_previsores = self.previsores.columns
        
    def padronizarDados(self):
        self.previsores = self.scaler.fit_transform(self.previsores)
        
    def preProcessar(self, configs):
        for col in configs['cols_categoria_nominal']:
            self.transformarVariavelCategoriaNominal(col)
            
        for col in configs['cols_dummy']:
            self.transformarVariavelDummy(col)
        
        if configs['padronizacao']:
            self.padronizarDados()
            
        self.previsores_treinamento, self.previsores_teste, self.classe_treinamento, self.classe_teste = train_test_split(self.previsores, self.classe, test_size=0.25, random_state=0)
        
        
    