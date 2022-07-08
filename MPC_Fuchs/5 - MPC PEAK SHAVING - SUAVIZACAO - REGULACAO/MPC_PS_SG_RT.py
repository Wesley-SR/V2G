#==================================================================================================================================================#
# MPC SUAVIZAÇÃO DA GERAÇÃO - MICROGRID COPEL
# Última Atualização: 27 de Maio de 2021
# Otimização utilizando MILP com a biblioteca PulP (Solver: CBC)
# Suavização realizada com base na aplicação do filtro Savitzky-Golay
#==================================================================================================================================================#

import csv
from funcao_criar_med_PS_SG_RT import criar_medicoes_PS_SG_RT
from fazer_previsao_PS_SG_RT import previsao_PS_SG_RT
from otimizacao_PS_SG_RT import otimizacao_PS_SG_RT

P = []        # Matriz para salvar os resultados da otimização
M = []        # Matriz com os valores totais dos dados de entrada
janela = 144  # Horizinte de previsão (144 = 24h)
periodo = 2 # Tempo em que é rodado o MPC (144 = 1 dia de operação)
ts = 0.16667

#==================================================================================================================================================#
# CRIAÇÃO DA MATRIZ N e Resultado
# O que é a Matriz N: Matriz com os dados de previsão para uma determinada janela de previsão
# N[143][3]: Tamanho: 144 Linhas; 4 Colunas (Tamanho da matriz M para uma janela de previsão de 24h)
N = []
for p in range(0,janela):
    N.append([0]*5)

# O que é a Matriz resultado: Matriz com os resultados das açoes de controle ao longo do periodo analisado
resultado = []
for d in range(0,periodo):
    resultado.append([0]*11)

#==================================================================================================================================================#
# LEITURA DOS DADOS HISTÓRICOS
# O que é a Matriz M: Matriz com TODAS as previsoes ao longo de um determinado periodo
# MATRIZ M[287][3]: Tamanho: 288 Linhas; 4 Colunas
dados_previsao_total = open('Curva_carga_dia_18_10_2017_nublado_01.csv','r')
leitor = csv.reader(dados_previsao_total)
next(leitor)
for linha in leitor:
    M.append(linha)
dados_previsao_total.close()

#==================================================================================================================================================#
for i in range(0,periodo):
    
    cont = 0
    for j in range(i, janela + i):
        for k in range(0, 5):
            N[cont][k] = M[j][k]
        cont = cont + 1
    criar_medicoes_PS_SG_RT(janela,N)
    
    previsao_PS_SG_RT() 
    
    #==============================================================================================================================================#
    # ATUALIZAR O ESTADO DE CARGA
    
    estado_de_carga = open('SoC_atualizado_PS_SG_RT.csv', 'w')
    writer = csv.writer(estado_de_carga, lineterminator='\n')
    writer.writerow(('tempo1', 'SoCini'))
    if i == 0:
        SoCini = 0.5
        tempo1 = 0
        SoC = SoCini
    else:
        #SoC = SoC - ((float(resultado[i-1][1]) + float(resultado[i-1][0]))*0.16667)/560
        SoC = SoC + ((-float(resultado[i-1][0])*ts*(1/0.96))/560) + (((float(resultado[i-1][1])*ts)*0.96)/560)
    writer.writerow((tempo1,SoC))
    estado_de_carga.close()
    
    #==============================================================================================================================================#    
    P_dc_bat_atual = open('P_SAE_DES.csv', 'w')
    writer = csv.writer(P_dc_bat_atual, lineterminator='\n')    
    writer.writerow(('tempo1', 'P_SAE_DC'))
    if i == 0:
        P_SAE_DC = 0
        tempo2 = 0
    else:
        P_SAE_DC = float(resultado[i-1][0])
    
    writer.writerow((tempo2,P_SAE_DC))
    P_dc_bat_atual.close()    
    
    
    P_ch_bat_atual = open('P_SAE_CAR.csv', 'w')
    writer = csv.writer(P_ch_bat_atual, lineterminator='\n')    
    writer.writerow(('tempo1', 'P_SAE_CAR'))
    if i == 0:
        P_SAE_CAR = 0
        tempo2 = 0
    else:
        P_SAE_CAR = float(resultado[i-1][1])
    
    writer.writerow((tempo2,P_SAE_CAR))
    P_ch_bat_atual.close() 
    #==============================================================================================================================================#
    # RESOLVER O PROBLEMA DE OTIMIZAÇÃO
    
    otimizacao_PS_SG_RT()
    resultados_puLP = open('resultado_otimizacao_PS_SG_RT.csv', 'r')
    leitor1 = csv.reader(resultados_puLP)
    next(leitor1)
    P.clear()
    for linha1 in leitor1:
        P.append(linha1)
    resultados_puLP.close()

    for c in range(0, 10):
        resultado[i][c] = P[0][c]
    
    print(i)

resultado_MPC = open('resultado_MPC_python_PS_SG_RT.csv', 'w')
writer = csv.writer(resultado_MPC, lineterminator='\n')
writer.writerow(('Bateria_descarregar', 'Bateria_carregar', 'potencia_REDE', 'SoC', 'PV_ref','custo_energia', 'Q_rede', 'Q_bat_PU', 'P_bat_PU', 'V_PCC'))
for g in range(0, periodo):
    writer.writerow((resultado[g][0], resultado[g][1], resultado[g][2], resultado[g][3], resultado[g][4], resultado[g][5], resultado[g][6], resultado[g][7], resultado[g][8], resultado[g][9]))
resultado_MPC.close()

