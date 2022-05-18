#================================================================================================#

# PROJETO V2G

# MODELO DE OTIMIZAÇÃO 1 - MILP

#================================================================================================#

import pulp as pl
import pandas as pd

dados_entrada = pd.read_csv('dados_entrada.csv', index_col=['tempo'])

prob = pl.LpProblem("otimizacao_V2G", pl.LpMinimize)

# VARIÁVEIS DO PROBLEMA
P_rede = pl.LpVariable.dicts('P_rede', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=100, cat='Continuous')

P_ch_bike = pl.LpVariable.dicts('P_ch', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=3, cat='Continuous')  
P_dc_bike = pl.LpVariable.dicts('P_dc', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=3, cat='Continuous')

P_ch_bat_est = pl.LpVariable.dicts('P_ch_bat_est', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=3, cat='Continuous')  
P_dc_bat_est = pl.LpVariable.dicts('P_dc_bat_est', ((tempo) for tempo in dados_entrada.index), lowBound=0, upBound=3, cat='Continuous')

Carga_bat_est = pl.LpVariable.dicts('Carga_bat_est', ((tempo) for tempo in dados_entrada.index), cat='Binary')
Descarga_bat_est = pl.LpVariable.dicts('Descarga_bat_est', ((tempo) for tempo in dados_entrada.index), cat='Binary')

Carga_bat_bike = pl.LpVariable.dicts('Carga_bat_bike', ((tempo) for tempo in dados_entrada.index), cat='Binary')
Descarga_bat_bike = pl.LpVariable.dicts('Descarga_bat_bike', ((tempo) for tempo in dados_entrada.index), cat='Binary')

SOC_bike = pl.LpVariable.dicts('SOC_bike', ((tempo) for tempo in dados_entrada.index), lowBound=0.2, upBound=1, cat='Continuous')
SOC_est = pl.LpVariable.dicts('SOC_est', ((tempo) for tempo in dados_entrada.index), lowBound=0.2, upBound=1, cat='Continuous')

#======================= PARÂMETROS =============================

SOC_max = 1  # SOC Maximo da bateria
SOC_min = 0.2  # SOC minimo da bateria

SOC_ini_bike = 1
SOC_ini_est = 0.5

ts = 1  # Intervalo de tempo (0.16667h = 10 min)
E_bike = 560  # Energia total das baterias das bicicletas (kWh)


prob += pl.lpSum([((P_rede[i]*ts*dados_entrada.loc[i,'custo_energia'])) for i in dados_entrada.index])

P_max_bike = 0.5
P_max_bat_est = 3

E_bat_est = 2.4
E_bat_bike = 2.4

            
#======================= RESTRIÇÕES =============================

for i in dados_entrada.index:
    
    # BATERIA ESTACIONÁRIA:
    prob += P_dc_bat_est[i] == Descarga_bat_est[i] * P_max_bat_est     
    prob += P_ch_bat_est[i] == Carga_bat_est[i] * P_max_bat_est        
    prob += Descarga_bat_est[i] + Carga_bat_est[i] <= 1
    
    
    # BATERIA BIKE:
    prob += P_dc_bike[i] == Descarga_bat_bike[i] * P_max_bike     
    prob += P_ch_bike[i] == Carga_bat_bike[i] * P_max_bike       
    prob += Descarga_bat_bike[i] + Carga_bat_bike[i] <= 1

    prob += P_dc_bike[i] + dados_entrada.loc[i,'potencia_PV'] + P_dc_bat_est[i] + P_rede[i] == P_ch_bike[i] + P_ch_bat_est[i] 

    # Calculo do Estado de Carga (SOC) - Baterias Bike
    
    if i == 0:
        prob += SOC_bike[i] == SOC_ini_bike
        prob += SOC_est[i] == SOC_ini_est
    else:
        prob += SOC_bike[i] == (SOC_bike[i-1] + ((-P_dc_bike[i-1]*ts)/E_bat_bike) + ((P_ch_bike[i-1]*ts)/E_bat_bike))
        prob += SOC_est[i] == (SOC_est[i-1] + ((-P_dc_bat_est[i-1]*ts)/E_bat_est) + ((P_ch_bat_est[i-1]*ts)/E_bat_est))
   
prob.solve()

P_rede_res = [0.0] * 25
P_ch_bike_res = [0.0] * 25
P_dc_bike_res = [0.0] * 25
P_ch_bat_est_res = [0.0] * 25
P_dc_bat_est_res = [0.0] * 25

for i in dados_entrada.index:
        var_output1 = P_rede[i].varValue
        P_rede_res[i] = var_output1
        
        var_output2 = P_ch_bike[i].varValue
        P_ch_bike_res[i] = var_output2

        var_output3 = P_dc_bike[i].varValue
        P_dc_bike_res[i] = var_output3
        
        var_output4 = P_ch_bat_est[i].varValue
        P_ch_bat_est_res[i] = var_output4
        
        var_output5 = P_dc_bat_est[i].varValue
        P_dc_bat_est_res[i] = var_output5








