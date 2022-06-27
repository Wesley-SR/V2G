def previsao_PS_SG_RT():
    import csv
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from numpy import array
    from keras.models import load_model
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    from keras.models import model_from_json
    import pandas as pd
    
    U = []
    medicoes = open('medicoes_PS_SG_RT.csv', 'r')
    leitor = csv.reader(medicoes)
    next(leitor)
    for linha in leitor:
        U.append(linha)
    medicoes.close()
    
    dados_medicoes_10min = open('medicoes_10_min_PS_SG_RT.csv', 'w')
    writer = csv.writer(dados_medicoes_10min, lineterminator='\n')
    writer.writerow(('tempo', 'potencia_ativa', 'potencia_reativa', 'potencia_PV', 'custo_energia'))
    
    dados_10_min_PV = []
    for p in range(0, 144):
        dados_10_min_PV.append([0])
    
    dados_10_min_carga = []
    for p in range(0, 144):
        dados_10_min_carga.append([0])
    
    dados_10_min_reativo = []
    for p in range(0, 144):
        dados_10_min_reativo.append([0.0])
    
    y = 0
    for l in range(0, 144):
        U[l][0] = y
        writer.writerow((U[l][0], U[l][1], U[l][2], U[l][3], U[l][4]))
        dados_10_min_PV[y] = U[l][3]
        dados_10_min_carga[y] = U[l][1]
        dados_10_min_reativo[y] = U[l][2]
        y = y + 1
    
    dados_medicoes_10min.close()
    
    dados_10_min_carga = [float(i) for i in dados_10_min_carga]  # Transforma todos os elementos str em float
    dados_10_min_carga = np.array([dados_10_min_carga])  # Transforma lista em vetor
    dados_10_min_carga = dados_10_min_carga.T
    
    dados_10_min_reativo = [float(i) for i in dados_10_min_reativo]  # Transforma todos os elementos str em float
    dados_10_min_reativo = np.array([dados_10_min_reativo])  # Transforma lista em vetor
    dados_10_min_reativo = dados_10_min_reativo.T
    
    dados_10_min_PV = [float(i) for i in dados_10_min_PV]
    dados_10_min_PV = np.array([dados_10_min_PV])
    dados_10_min_PV = dados_10_min_PV.T
       
      
    #==================================================================================================================================================#
    # ALGORITMO DE PREVISÃO
    
    # PREVISÃO CARGA ATIVA:
    arquivo_1 = open('previsao_carga_ativa_10_min.json', 'r')
    estrutura_rede_carga = arquivo_1.read()
    arquivo_1.close() # Para liberar memória
    classificador_carga = model_from_json(estrutura_rede_carga)
    classificador_carga.load_weights('LSTM_Carga_Ativa_144.h5')
        
    
    n_steps_in = 144
    n_features = 1
    
    dados_10_min_carga = dados_10_min_carga
    
    # Normalizar os dados de entrada:
    scaler = MinMaxScaler(feature_range=(0, 1))
    carga = scaler.fit_transform(dados_10_min_carga)
    
    x_input_1 = [i[0] for i in carga.tolist()]  # Transforma os valores do dataset em uma lista
    x_input_1 = array(x_input_1)
    x_input_1 = x_input_1.reshape((1, n_steps_in, n_features))
    previsao_carga = classificador_carga.predict(x_input_1, verbose=0)
    
    previsao_carga = scaler.inverse_transform(previsao_carga)
    previsao_carga = previsao_carga
    
    
    
    # PREVISÃO CARGA REATIVA:
    arquivo_3 = open('previsao_carga_reativa_10_min.json', 'r')
    estrutura_rede_carga_reativa = arquivo_3.read()
    arquivo_3.close() # Para liberar memória
    classificador_carga_reativa = model_from_json(estrutura_rede_carga_reativa)
    classificador_carga_reativa.load_weights('LSTM_Carga_Reativa_144.h5')
        
    dados_10_min_reativo = dados_10_min_reativo
    
    # Normalizar os dados de entrada:
    scaler = MinMaxScaler(feature_range=(0, 1))
    reativo = scaler.fit_transform(dados_10_min_reativo)
    
    x_input_3 = [i[0] for i in reativo.tolist()]  # Transforma os valores do dataset em uma lista
    x_input_3 = array(x_input_3)
    x_input_3 = x_input_3.reshape((1, n_steps_in, n_features))
    previsao_carga_reativa = classificador_carga_reativa.predict(x_input_3, verbose=0)
    
    previsao_carga_reativa = scaler.inverse_transform(previsao_carga_reativa)
    previsao_carga_reativa = previsao_carga_reativa
    
    
    # PREVISÃO PV:
    arquivo_2 = open('previsao_PV_10_min.json', 'r')
    estrutura_rede_PV = arquivo_2.read()
    arquivo_2.close() # Para liberar memória
    classificador_PV = model_from_json(estrutura_rede_PV)
    classificador_PV.load_weights('LSTM_PV_144_2.h5') 
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    PV = scaler.fit_transform(dados_10_min_PV)
    
    # demonstrate prediction
    x_input_2 = [i[0] for i in PV.tolist()]  # Transforma os valores do dataset em uma lista
    x_input_2 = array(x_input_2)
    x_input_2 = x_input_2.reshape((1, n_steps_in, n_features))
    previsao_PV = classificador_PV.predict(x_input_2, verbose=0)
    
    previsao_PV = scaler.inverse_transform(previsao_PV)
    
    
    #==================================================================================================================================================#
    # CRIAR CURVA DE REFERÊNCIA UTILIZANDO MÉDIA MÓVEL
    
    y_carga = previsao_carga
    y_PV = previsao_PV
    y_reativo = previsao_carga_reativa
    
    
    curva_PV = previsao_PV
     
    curva_PV = curva_PV.reshape(144).tolist()
    
    curva_PV.insert(0,dados_10_min_PV[143][0])
    curva_PV.insert(0,dados_10_min_PV[142][0])
    curva_PV.insert(0,dados_10_min_PV[141][0])
    curva_PV.insert(0,dados_10_min_PV[140][0])
    curva_PV.insert(0,dados_10_min_PV[139][0])
    curva_PV.insert(0,dados_10_min_PV[138][0])
    curva_PV.insert(0,dados_10_min_PV[137][0])
    curva_PV.insert(0,dados_10_min_PV[136][0])
    curva_PV.insert(0,dados_10_min_PV[135][0])
    
    window_size_1 = 10
    numbers_series_1 = pd.Series(curva_PV)
    windows1 = numbers_series_1.rolling(window_size_1)
    moving_averages_1 = windows1.mean()
    curva_ref = np.array(moving_averages_1,dtype = float)
    
    indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    curva_ref = np.append(curva_ref, ([0.0],[0.0],[0.0],[0.0],[0.0]))
    
    
    curva_ref = np.delete(curva_ref, indexes)
    curva_ref = curva_ref.reshape(144)  # Transforma [[]] em [] 
        
    #==================================================================================================================================================#
    # ARQUIVO PARA A OTIMIZAÇÃO
    
    # ------------------------ FORMATO DO ARQUIVO -------------------------------------------------------------#
    # 'tempo' , 'potencia_ativa_prevista', 'potencia_reativa_prevista', 'potencia_PV_prevista', 'custo_energia'
    #    0    ,       x [kW]  ,                     y[kVAr]                  ,   a[kW]              z[$/kWh]
    #    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #   143   ,       x [kW]  ,                     y[kVAr]                  ,   a[kW]              z[$/kWh]
    # ---------------------------------------------------------------------------------------------------------#
    
    y_carga = y_carga.reshape(144, 1)
    y_PV = y_PV.reshape(144, 1)
    y_reativo = y_reativo.reshape(144,1)
    
    vet_tempo_aux = []
    for p in range(0, 144):
        vet_tempo_aux.append([0])
    
    especificacoes = open('especificacoes_2_PS_SG_RT.csv', 'w')
    writer = csv.writer(especificacoes, lineterminator='\n')
    writer.writerow(('tempo', 'potencia_ativa_prevista', 'potencia_reativa_prevista', 'potencia_PV_prevista', 'custo_energia', 'PV_ref'))
    writer.writerow((vet_tempo_aux[0][0], dados_10_min_carga[143][0], dados_10_min_reativo[143][0], dados_10_min_PV[143][0], U[0][4], curva_ref[0]))
    y = 1
    for l in range(1, 144):
        vet_tempo_aux[l][0] = y
        writer.writerow((vet_tempo_aux[l][0], y_carga[l][0], y_reativo[l][0], y_PV[l][0], U[l][4], curva_ref[l]))
        y = y + 1
    especificacoes.close()   

    




























