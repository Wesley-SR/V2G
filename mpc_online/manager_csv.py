#!/usr/bin/env python3

# Esse código lê os dados de um CSV e escreve na rede Modbus
# Fonte:
    # https://www.youtube.com/watch?v=FYPQgnQE9fk&ab_channel=Johannes4GNU_Linux
    # https://github.com/Johannes4Linux/Simple-ModbusTCP-Server/blob/master/Simple_ModbusServer.py
    # https://pymodbustcp.readthedocs.io/en/latest/examples/server.html?highlight=server
    # https://pymodbustcp.readthedocs.io/en/latest/package/class_ModbusServer.html?highlight=server#class-modbusserver

# read_register
# read 10 registers and print result on stdout

import time

from pyModbusTCP.client import ModbusClient
import pandas as pd

if __name__ == '__main__':

    # caminho = "C:\Users\wesle\Dropbox\Lactec\DADOS\MPC_MATLAB_Peak_shaving_Suavizacao_Regulacao\Curva_carga_dia_08_11_17.csv"
    caminho_do_arquivo = "dados_entrada.csv"
    
    # Faz a leitura do arquivo
    medidas_csv = pd.read_csv(caminho_do_arquivo, sep = ";", index_col=['tempo'])
    
    # init modbus client
    # host = '192.168.0.7'
    host = '127.1.0.0'
    port = 502
    client_ID = 2
    
    try:
        c = ModbusClient(host = host, port=port, unit_id = client_ID, debug=False, auto_open=True)
    except ValueError:
        print("Error with host or port params")
    
    # main read loop
    contador = 1
    somar = 1
    
    # Alimentador das medições do P_ativa_carga (P), P_reativa_carga (Q) e P_pv (PV)
    while True:
        
        # Lê a linha atual (no caso a 144) a partir da linha 144 (substituir por medições online)
        N = medidas_csv.iloc[contador].values
        # M = [int(N[0]), int(N[1]), int(N[2]), int(N[3])] # Amostra, P, Q e PV
        # amostra = M[0]
        # P_ativa = M[1]
        # P_reativa = M[2]
        # P_pv = M[3]
        
        time.sleep(2)
        try:
            
            # Leitura
            print("\n\n")
            registers = c.read_holding_registers(0, 23)
            # if success display registers
            if registers:
                print("Read: \n {}".format(registers))
            else:
                print('unable to read registers \n')            

            # Escrita
            print("Write: \n {}".format(N))
            c.write_multiple_registers(0,N)
            
            # Leitura
            registers = c.read_holding_registers(0, 23)
            # if success display registers
            if registers:
                print("Read: \n {}".format(registers))
            else:
                print('unable to read registers \n')

            contador += 1
            if contador > 287:
                contador = 1
                
        except Exception as error:
            print("{}\n".format(error))
            c.close()
            break
    
    print("Cliente encerrado")