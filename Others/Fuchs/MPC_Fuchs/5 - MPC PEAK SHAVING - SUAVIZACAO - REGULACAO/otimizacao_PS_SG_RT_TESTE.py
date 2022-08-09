#=================================================================================================================================================#
 # IMPORTAÇÃO DAS BIBLIOTECAS NECESSARIAS 
 
import pandas as pd  # Manipulação, Leitura e Visualização de Dados
import pulp as pl    # Ferramenta de Programação Linear
import csv

 #=================================================================================================================================================#
 # DADOS DE ENTRADA 
 
dados_entrada = pd.read_csv('especificacoes_2_PS_SG_RT.csv', index_col=['tempo'])
prob = pl.LpProblem("Peak_Sahving_Suavizacao_Regulacao", pl.LpMinimize)  # Declaração do modelo

#=================================================================================================================================================#
# DECLARAÇÃO DA VARIÁVEIS DO PROBLEMA 

Q_bat = pl.LpVariable.dicts('Q_bat', ((tempo) for tempo in dados_entrada.index), lowBound=-0.6, upBound=0.6, cat='Continuous')

P_bat = pl.LpVariable.dicts('P_bat', ((tempo) for tempo in dados_entrada.index), lowBound=-1, upBound=1, cat='Continuous')

V_PCC = pl.LpVariable.dicts('V_PCC', ((tempo) for tempo in dados_entrada.index), cat='Continuous')

t1_reg = pl.LpVariable.dicts('t1_reg', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')
t2_reg = pl.LpVariable.dicts('t2_reg', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')

P_rede = pl.LpVariable.dicts("P_rede", ((tempo) for tempo in dados_entrada.index), lowBound=0, cat='Continuous')

q_rede = pl.LpVariable.dicts("q_rede",((tempo) for tempo in dados_entrada.index), cat= 'Continuous')



P_ch = pl.LpVariable.dicts('P_ch', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=250, cat='Continuous')

P_dc = pl.LpVariable.dicts('P_dc', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=250, cat='Continuous')

delta_ch = pl.LpVariable.dicts('delta_ch', ((tempo) for tempo in dados_entrada.index), cat='Binary')

delta_dc = pl.LpVariable.dicts('delta_dc', ((tempo) for tempo in dados_entrada.index), cat='Binary')

SOC = pl.LpVariable.dicts('SOC', ((tempo) for tempo in dados_entrada.index), lowBound=0.2, upBound=0.8, cat='Continuous')

delta_M = pl.LpVariable.dicts('delta_M',((tempo) for tempo in dados_entrada.index), cat='Binary')

delta_NM = pl.LpVariable.dicts('delta_NM',((tempo) for tempo in dados_entrada.index), cat='Binary')

multa = pl.LpVariable.dicts('multa', ((tempo) for tempo in dados_entrada.index), cat='Continuous')

t1 = pl.LpVariable.dicts('t1', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')
t2 = pl.LpVariable.dicts('t2', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')

t1_soc = pl.LpVariable.dicts('t1_soc', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')
t2_soc = pl.LpVariable.dicts('t2_soc', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')

var_dc = pl.LpVariable.dicts('var_dc',((tempo) for tempo in dados_entrada.index), cat= 'Continuous')
var_ch = pl.LpVariable.dicts('var_ch',((tempo) for tempo in dados_entrada.index), cat= 'Continuous')

x1 = pl.LpVariable.dicts('x1', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')
x2 = pl.LpVariable.dicts('x2', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')    

y1 = pl.LpVariable.dicts('y1', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')
y2 = pl.LpVariable.dicts('y2', ((tempo) for tempo in dados_entrada.index), lowBound= 0, cat='Continuous')    

 #=================================================================================================================================================#
 # PARÂMETROS
 
d = 0.54  # Custo utilização bateria (R$/kWh)
SOC_max = 0.8  # SOC Maximo da bateria
SOC_min = 0.2  # SOC minimo da bateria
ts = 0.16667  # Intervalo de tempo (0.16667h = 10 min)
E = 560  # Energia total da bateria (kWh)
P_max = 250  # Potencia maxima de carga e descarga da bateria
U         = 3000    # Parametro da formulação Big-M
L         = -3000   # Parametro da formulação Big-M
epsilion  = 0.01    # Parametro da formulação Big-M
lim       = 300     # Limite de potencia demandada pela subestação (kW)
SOC_ref   = 0.8
 
V_PCC_ref = 1         # Tensão de referência no PCC
K = 13094.95 # Fator de Sensibilidade no PCC (kVar/pu) 
 
 
SOC_ini = (pd.read_csv('SoC_atualizado_PS_SG_RT.csv', index_col=['tempo1']))
SOC_ini = float(SOC_ini.loc[0])

P_SAE_DC_ANT = (pd.read_csv('P_SAE_DES.csv', index_col=['tempo1']))
P_SAE_DC_ANT = float(P_SAE_DC_ANT.loc[0])

P_SAE_CH_ANT = (pd.read_csv('P_SAE_CAR.csv', index_col=['tempo1']))
P_SAE_CH_ANT = float(P_SAE_CH_ANT.loc[0])   
 
 #=================================================================================================================================================#
 # FUNÇÃO OBJETIVO

 
 # PEAK SHAVING + SUAVIZAÇÃO DA GERAÇÃO + REGULAÇÃO DE TENSÃO:
 
prob += pl.lpSum([(0.3*(P_rede[i]*ts*dados_entrada.loc[i,'custo_energia']*(1/80))) + 
             (0.1*((P_dc[i]+P_ch[i])*ts*d*(1/22.5))) +
             (0.5*(multa[i]*1/50)) +
             (0.01*(t1_soc[i]+t2_soc[i])) +
             (0.1*((t1[i]+t2[i])*(1/60))) +
             (t1_reg[i]+t2_reg[i]) for i in dados_entrada.index])
   
 # Todos peso 1 e multa 10
 #=================================================================================================================================================#
 # RESTRIÇÕES
 
for i in dados_entrada.index:
     
    prob += P_dc[i] <= delta_dc[i] * P_max
    
    prob += P_ch[i] <= delta_ch[i] * P_max
    
    prob += delta_ch[i] + delta_dc[i] <= 1
   
    prob += P_dc[i] + P_rede[i] + dados_entrada.loc[i,'potencia_PV_prevista'] == P_ch[i] + dados_entrada.loc[i,'potencia_ativa_prevista']
               
    prob += SOC[i] <= SOC_max
    
    prob += SOC[i] >= SOC_min
     
     
    if dados_entrada.loc[i, 'PV_ref'] > 0:
        prob += t1[i] - t2[i] == ((dados_entrada.loc[i, 'PV_ref']) - (dados_entrada.loc[i, 'potencia_PV_prevista'] + P_dc[i] - P_ch[i]))
    #else:
        #prob += t1[i] - t2[i] == 0
        
    # Calculo do Estado de Carga (SOC) 
    if i == 0:
        prob += SOC[i] == SOC_ini
    else:
        prob += SOC[i] == SOC[i-1] + ((-P_dc[i-1]*(1/0.96)*ts)/E) + ((0.96*P_ch[i-1]*ts)/E)
                     
         
    # Condições para a aplicação da penalidade caso lim > 226 kW
    prob += P_rede[i] - lim >= L*(1-delta_M[i])
    prob += P_rede[i] - lim <= (U+epsilion)*delta_M[i] - epsilion
    prob += multa[i] - 50 <= U*(1 - delta_M[i])
    prob += multa[i] - 50 >= L*(1 - delta_M[i])
    prob += delta_M[i] + delta_NM[i] == 1
    prob += multa[i] <= U*(1 - delta_NM[i])
    prob += multa[i] >= L*(1 - delta_NM[i])

    prob += t1_soc[i] - t2_soc[i] == (SOC_ref - SOC[i]) 
                        

    # Deixar nessa ordem:
    if i == 0:
        if P_dc[i] >= P_SAE_DC_ANT:
            prob += P_dc[i] - P_SAE_DC_ANT <= 100
        else:
            prob += P_SAE_DC_ANT - P_dc[i] <= 100 

    if i == 0:
        if P_ch[i] >= P_SAE_CH_ANT:
            prob += P_ch[i] - P_SAE_CH_ANT <= 100
        else:
            prob += P_SAE_CH_ANT - P_ch[i] <= 100                
 
    prob += V_PCC[i] == (-(1/K)*(dados_entrada.loc[i, 'potencia_reativa_prevista'] - Q_bat[i])) + 0.9955
    
    prob += t1_reg[i] - t2_reg[i] == (V_PCC_ref - V_PCC[i])
    
 #------------- CURVA CAPABILIDADE INVERSOR LINEARIZADA -------------------#
    prob += Q_bat[i] <= (-0.99*P_bat[i] + 1)/(0.14106736)
    prob += Q_bat[i] <= (-0.792*P_bat[i] + 1)/(0.610521089)
    prob += Q_bat[i] <= (-0.594*P_bat[i] + 1)/(0.80446504)
    prob += Q_bat[i] <= (-0.396*P_bat[i] + 1)/(0.91825051)
    prob += Q_bat[i] <= (-0.198*P_bat[i] + 1)/(0.98020202)
    prob += Q_bat[i] <= (0*P_bat[i] + 1)/(1)
    prob += Q_bat[i] <= (0.198*P_bat[i] + 1)/(0.98020202)
    prob += Q_bat[i] <= (0.396*P_bat[i] + 1)/(0.91825051)
    prob += Q_bat[i] <= (0.594*P_bat[i] + 1)/(0.80446504)
    prob += Q_bat[i] <= (0.792*P_bat[i] + 1)/(0.610521089)
    prob += Q_bat[i] <= (0.99*P_bat[i] + 1)/(0.14106736)
    
    prob += Q_bat[i] >= -((-0.99*P_bat[i] + 1)/(0.14106736))
    prob += Q_bat[i] >= -((-0.792*P_bat[i] + 1)/(0.610521089))
    prob += Q_bat[i] >= -((-0.594*P_bat[i] + 1)/(0.80446504))
    prob += Q_bat[i] >= -((-0.396*P_bat[i] + 1)/(0.91825051))
    prob += Q_bat[i] >= -((-0.198*P_bat[i] + 1)/(0.98020202))
    prob += Q_bat[i] >= -((0*P_bat[i] + 1)/(1))
    prob += Q_bat[i] >= -((0.198*P_bat[i] + 1)/(0.98020202))
    prob += Q_bat[i] >= -((0.396*P_bat[i] + 1)/(0.91825051))
    prob += Q_bat[i] >= -((0.594*P_bat[i] + 1)/(0.80446504))
    prob += Q_bat[i] >= -((0.792*P_bat[i] + 1)/(0.610521089))
    prob += Q_bat[i] >= -((0.99*P_bat[i] + 1)/(0.14106736))   
    
    prob += P_bat[i] == P_dc[i]*(1/250) - P_ch[i]*(1/250)


    # Balanço de potência reativa:
    prob += q_rede[i] + (Q_bat[i]*250) == dados_entrada.loc[i,'potencia_reativa_prevista']   

 #=================================================================================================================================================#
 # RESOLVER O PROBLEMA
 
prob.solve()
 
 #=================================================================================================================================================#
 # MOSTRAR RESULTADOS
 
P_dc_res = [0.0] * 144
P_ch_res = [0.0] * 144
SOC_res = [0.0] * 144
P_rede_res = [0.0] * 144

Q_rede_res = [0.0] * 144   
Q_bat_res = [0.0] * 144    
V_PCC_res = [0.0] * 144 
P_bat_res = [0.0] * 144   
        

for i in dados_entrada.index:
    var_output1 = P_dc[i].varValue
    P_dc_res[i] = var_output1

    var_output2 = P_ch[i].varValue
    P_ch_res[i] = var_output2

    var_output3 = SOC[i].varValue
    SOC_res[i] = var_output3

    var_output4 = P_rede[i].varValue
    P_rede_res[i] = var_output4
    
    var_output5 = q_rede[i].varValue
    Q_rede_res[i] = var_output5

    var_output6 = Q_bat[i].varValue
    Q_bat_res[i] = var_output6

    var_output7 = V_PCC[i].varValue
    V_PCC_res[i] = var_output7 
    
    var_output8 = P_bat[i].varValue
    P_bat_res[i] = var_output8 
    
       
    Referencia_PV = dados_entrada.iloc[:,4]
    
    Referencia_PV = Referencia_PV.values.tolist()
    
    custo_energia = dados_entrada.iloc[:,3]
    custo_energia = custo_energia.values.tolist()
 
 #=================================================================================================================================================#
 # GERAR ARQUIVO .CSV COM AS RESPOSTAS
 
resultado_otimizacao = open('resultado_otimizacao_PS_SG_RT.csv', 'w')
writer = csv.writer(resultado_otimizacao, lineterminator='\n')
writer.writerow(('Pot_descarga_bateria', 'Pot_carga_bateria', 'potencia_demandada_REDE', 'SOC','P_ref', 'custo_energia', 'Q_rede', 'Q_bat_PU', 'P_bat_PU', 'V_PCC'))
for g in range(0, 144):
    writer.writerow((P_dc_res[g], P_ch_res[g], P_rede_res[g], SOC_res[g], Referencia_PV[g], custo_energia[g], Q_rede_res[g], Q_bat_res[g], P_bat_res[g], V_PCC_res[g]))
resultado_otimizacao.close()






















