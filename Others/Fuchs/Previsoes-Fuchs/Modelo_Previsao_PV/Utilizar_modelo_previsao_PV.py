import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from keras.models import load_model
from scipy.interpolate import interp1d
from keras.models import model_from_json
import matplotlib.pyplot as plt


#------------------------- PREVISÃO CARGA COM DADOS DE TESTE --------------------------------#

#========================================================================================================================#
# DADOS NO ARQUIVO .CSV
U = []
dados_entrada_ativa = open('dados_teste_1_semana_PV.csv', 'r')
leitor = csv.reader(dados_entrada_ativa)
next(leitor)
for linha in leitor:
    U.append(linha)
dados_entrada_ativa.close()

# DADOS PV DIA 13:
dia_13_PV = []
for p in range(0, 144):
    dia_13_PV.append([0])
    dia_13_PV[p] = U[p][1] 
dia_13_PV = np.array([dia_13_PV]).T  


#========================================================================================================================#
# CARREGAR MODELO DE PREVISÃO E NORMALIZAR OS DADOS DE ENTRADA:

arquivo_2 = open('previsao_PV_10_min.json', 'r')
estrutura_rede_PV = arquivo_2.read()
arquivo_2.close() 
classificador_PV = model_from_json(estrutura_rede_PV)
classificador_PV.load_weights('LSTM_PV_144_2.h5') 

n_steps_in = 144
n_features = 1

# Normalizar os dados de entrada:
scaler = MinMaxScaler(feature_range=(0, 1))
PV_dia_13 = scaler.fit_transform(dia_13_PV)
 
# DEIXAR NA FORMA: (1,144,1)
x_input_1 = PV_dia_13.reshape((1, n_steps_in, n_features)) 

# FAZER A PREVISÃO:
previsao_PV_14 = classificador_PV.predict(x_input_1, verbose=0) 

# DEIXAR A PREVISÃO EM kW:
previsao_PV_14 = scaler.inverse_transform(previsao_PV_14).T


#========================================================================================================================#
# Comparar previsão com os dados reais:
# Dados reais do PV do dia 14:
dia_14_PV = []
for p in range(0, 144):
    dia_14_PV.append([0])
      
for l in range(144,288):
    dia_14_PV[l-144] = U[l][1]

dia_14_PV = np.array([float(i) for i in dia_14_PV]).T

plt.plot(dia_14_PV)
plt.plot(previsao_PV_14)
plt.xlabel('Instante de tempo')
plt.ylabel('Potência [kW]')
























