#==============================================================================================================================================================#
# LSTM - RNN para a previsão de Geração Solar - Faxinal do Céu - Microgrid COPEL
# Histórico de Dados: Medição de Geração Solar em Rosana-SP
# Intervalo entre as medições: 30 minutos

# Descrição da LSTM: A LSTM recebe como dados de entrada 24h de medições anteriores (48 amostras) e retorna a previsão para 24hs à frente (48 amostras)

#==============================================================================================================================================================#

# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs
import h5py
from numpy import array
from datetime import datetime
from keras.models import load_model
import h5py
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
#==============================================================================================================================================================#

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# CARREGAR O HISTÓRICO DE DADOS:
dataset = pd.read_csv('Dados_PV_10_min.csv', usecols=[1])
dataset = dataset.values
dataset = dataset.astype('float') # Transforma os valores em floot

# NORMALIZAR O HISTÓRICO DE DADOS: (deixar os valores das entradas entre 0 e 1):
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# DIVIDIR OS DADOS EM TREINO E TESTE
train_size = int(len(dataset)*0.85)   # 70% dos dados para treino + 15% para validação

test_size = len(dataset) - train_size # 15% dos dados para teste

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


raw_seq_train = [i[0] for i in train.tolist()] # Transforma os valores do dataset em uma lista
raw_seq_test = [i[0] for i in test.tolist()]   # Transforma os valores do dataset em uma lista

n_steps_in, n_steps_out = 144, 144

trainX, trainY = split_sequence(raw_seq_train, n_steps_in, n_steps_out)
testX, testY = split_sequence(raw_seq_test, n_steps_in, n_steps_out)

# DEIXAR A ENTRADA NA FORMA [samples, time steps, features]
n_features = 1
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))

#==============================================================================================================================================================#
#IMPLEMENTAÇÃO LSTM

model = Sequential()

# Primeira camada oculta:
model.add(LSTM(48, input_shape=(n_steps_in, n_features), return_sequences=True)) # Camada de entrada
# Segunda camada oculta:
model.add(Dropout(0.3))
# Terceira camada oculta:
model.add(LSTM(48)) 
model.add(Dropout(0.3))
# Quarta camada oculta:
#model.add(LSTM(2))

# Camada de saída:
model.add(Dense(n_steps_out, activation = 'relu'))

#==============================================================================================================================================================#
# TREINAMENTO DA LSTM:
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
history = model.fit(trainX, trainY, validation_split=0.20, epochs=80, verbose=2, batch_size = 5000)

loss_MSE_treino = history.history['loss']
loss_MSE_validacao = history.history['val_loss']

score = model.evaluate(trainX, trainY,verbose=2) # Avaliar o modelo com os dados de treino
model.summary()

print("Avaliação nos dados de Testes")
results = model.evaluate(testX, testY, verbose=1, return_dict=True) # Avaliar o modelo com os dados de teste
print("test loss mse, test mae:", results)
print(history.history.keys())

# PLOTAR LOSS (MSE)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['treino', 'Validação'], loc='upper left')
plt.show()

#==============================================================================================================================================================#
# FAZER PREVISÕES COM OS DADOS DE TESTE:
'''
# Vetor para Salvar as Previsões:
previsoes_dados_teste = []
for p in range(0, 8477):
    previsoes_dados_teste.append([0.0])    
previsoes_dados_teste = np.array(previsoes_dados_teste)  

# Vetor para salvar as entradas:
entradas_totais = []
for p in range(0, 8477):
    entradas_totais.append([0.0])    
entradas_totais = np.array(entradas_totais)  

# Vetor para salvar os valores reais:
valores_reais_totais = []
for p in range(0, 8477):
    valores_reais_totais.append([0.0])    
valores_reais_totais = np.array(valores_reais_totais)  

# Vetor para Salvar as Previsões:
previsoes_dados_teste_normalizada = []
for p in range(0, 8477):
    previsoes_dados_teste_normalizada.append([0.0])    
previsoes_dados_teste_normalizada = np.array(previsoes_dados_teste_normalizada)  

entradas_totais_normalizadas = []
for p in range(0, 8477):
    entradas_totais_normalizadas.append([0.0])    
entradas_totais_normalizadas = np.array(entradas_totais_normalizadas)  

valores_reais_totais_normalizados = []
for p in range(0, 8477):
    valores_reais_totais_normalizados.append([0.0])    
valores_reais_totais_normalizados = np.array(valores_reais_totais_normalizados)  

x = 0    
for j in range(0,175):
    
    entrada = test[0+x:48+x]
    
    valor_real = test[48+x:96+x]
    
    x_input_1 = [i[0] for i in entrada.tolist()] # Transforma os valores do dataset em uma lista
    
    x_input_1 = array(x_input_1)
    
    x_input_1 = x_input_1.reshape((1, n_steps_in, n_features))
    
    previsao_1 = model.predict(x_input_1, verbose=0)
    
    previsao_1 = previsao_1.T
    
    previsoes_dados_teste_normalizada[48+x:96+x] = previsao_1
    entradas_totais_normalizadas[0+x:48+x] = entrada
    valores_reais_totais_normalizados[48+x:96+x] = valor_real
    
    valor_real = scaler.inverse_transform(valor_real)
    previsao_1 = scaler.inverse_transform(previsao_1)
    
    entrada = scaler.inverse_transform(entrada)

    previsoes_dados_teste[48+x:96+x] = previsao_1
    entradas_totais[0+x:48+x] = entrada
    valores_reais_totais[48+x:96+x] = valor_real
    
    x = x + 48

plt.figure(2)
plt.plot(valores_reais_totais)
plt.plot(previsoes_dados_teste)
plt.ylabel('Potência (kW)')
plt.legend(['Valor Medido', 'Valor Previsto'], loc='upper left')
plt.ylim(ymax = 200, ymin = 0)
plt.show()
'''
#==============================================================================================================================================================#


# SALVAR O MODELO:
classificador_json = model.to_json()

with open('previsao_PV_10_min.json', 'w') as json_file:
    json_file.write(classificador_json)
model.save_weights('LSTM_PV_144_2.h5')
