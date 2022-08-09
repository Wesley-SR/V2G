import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import matplotlib.pyplot as plt


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
 
# DEIXAR NA FORMA: (1,144,1)
x_input_1 = PV_dia_25.reshape((1, n_steps_in, n_features)) 

# FAZER A PREVISÃO:
previsao_PV_26 = classificador_PV.predict(x_input_1, verbose=0) 

# DEIXAR A PREVISÃO EM kW:
previsao_PV_26 = scaler.inverse_transform(previsao_PV_26).T


#========================================================================================================================#
# Comparar previsão com os dados reais:
# Dados reais do PV do dia 14:
dia_26_PV = []
for p in range(0, 96):
    dia_26_PV.append([0])
      
for l in range(96,192):
    dia_26_PV[l-96] = U[l][1]

dia_26_PV = np.array([float(i) for i in dia_26_PV]).T

plt.plot(dia_26_PV)
plt.plot(previsao_PV_26)
plt.xlabel('Instante de tempo')
plt.ylabel('Potência [kW]')

''' WRITE RESULTS IN SCV'''
p_rede_df = pd.DataFrame(dia_26_PV, columns = ['p_pv'])
p_rede_df.to_csv("online_forecast_pv.csv", index=False)