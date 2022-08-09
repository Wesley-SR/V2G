import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import model_from_json


#------------------------- PREVISÃO CARGA COM DADOS DE TESTE ------------------------------------------------------------#

#========================================================================================================================#
# DADOS NO ARQUIVO .CSV
U = []
dados_entrada_ativa = open('dados_testes.csv', 'r')
leitor = csv.reader(dados_entrada_ativa)
next(leitor)
for linha in leitor:
    U.append(linha)
dados_entrada_ativa.close()

# DADOS PV DIA 25:
dia_25_PV = []
for p in range(0, 96):
    dia_25_PV.append([0])
    dia_25_PV[p] = U[p][1] 
dia_25_PV = np.array([dia_25_PV]).T  


#========================================================================================================================#
# CARREGAR MODELO DE PREVISÃO E NORMALIZAR OS DADOS DE ENTRADA:

arquivo_2 = open('previsao_PV_15_min.json', 'r')
estrutura_rede_PV = arquivo_2.read()
arquivo_2.close() 
classificador_PV = model_from_json(estrutura_rede_PV)
classificador_PV.load_weights('LSTM_PV_96.h5') 

n_steps_in = 96
n_features = 1


# FAZER PREVISÃO DO PRIMEIRO DIA:

# Normalizar os dados de entrada:
scaler = MinMaxScaler(feature_range=(0, 1))
PV_dia_25 = scaler.fit_transform(dia_25_PV)
 
# DEIXAR NA FORMA: (1,96,1)
x_input_1 = PV_dia_25.reshape((1, n_steps_in, n_features)) 

# FAZER A PREVISÃO DO PRIMEIRO DIA:
previsao_PV_26 = classificador_PV.predict(x_input_1, verbose=0) 

x_input_2 = previsao_PV_26.reshape((1, n_steps_in, n_features))

# FAZER A PREVISÃO DO SEGUNDO DIA:
previsao_PV_27 = classificador_PV.predict(x_input_2, verbose=0)


# DEIXAR A PREVISÃO EM kW:
previsao_PV_26 = scaler.inverse_transform(previsao_PV_26).T
previsao_PV_27 = scaler.inverse_transform(previsao_PV_27).T

# Previsão Total:
previsao = np.concatenate((previsao_PV_26,previsao_PV_27), axis = 0)

''' WRITE RESULTS IN SCV'''
p_rede_df = pd.DataFrame(previsao, columns = ['p_pv'])
p_rede_df.to_csv("offline_forecast_pv.csv", index=False)
