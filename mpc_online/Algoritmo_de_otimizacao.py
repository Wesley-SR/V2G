'''============================================================================
PROJETO V2G
MODELO DE OTIMIZAÇÃO 1 - MILP

Mudança em relação ao código anterior:
    - Voltou com o peso do soc_bateria_bike somando ao longo do somatório
    - Foi arrumado o peso da estacionária, pois estava subtraindo na FO
    - Os pesos agora variam ao longo do tempo (Tarefa do usuário)
Observações:
    - Aumentando o peso peso_soc_bike o algoritmo está muito lento para quando
    multiplicado por (-1) na FO (Travando para PULP_CBC_CMD)
#==========================================================================='''


import pulp as pl
import pandas as pd
import matplotlib.pyplot as plt

''' INPUT DATA '''
dados_entrada = pd.read_csv('dados_entrada_15_min.csv', index_col=['tempo'], sep=";")
dados_ref = pd.read_csv('p_rede_ref.csv', index_col=['tempo'], sep=";")

''' ------- DEFINIÇÃO DO PROBLEMA ------- '''

prob = pl.LpProblem("otimizacao_V2G", pl.LpMinimize)
solver = pl.SCIP_CMD()
# Solver Disponívels
# ['PULP_CBC_CMD', 'MIPCL_CMD', 'SCIP_CMD']
# Solver Possíveis
# ['GLPK_CMD', 'PYGLPK', 'CPLEX_CMD', 'CPLEX_PY', 'CPLEX_DLL', 'GUROBI', 
# 'GUROBI_CMD', 'MOSEK', 'XPRESS', 'PULP_CBC_CMD', 'COIN_CMD', 'COINMP_DLL', 
# 'CHOCO_CMD', 'MIPCL_CMD', 'SCIP_CMD']

time_array = dados_entrada.index

''' ------- PARÂMETROS ------- ''' 
ts = 0.25 # Intervalo de tempo (0.25h = 15 min)
num_amostras = time_array.size
# REDE
p_max_rede_imp = 20
p_max_rede_exp = 20
p_max_pv = 10
custo_max_energia = 1
# BATERIA BICICLETA
num_bikes = 10
soc_max_bike = 1 # soc Maximo da bateria
soc_min_bike = 0 # soc minimo da bateria
soc_ini_bike = 0.8
p_max_bat_bike = 0.5 # (kW) -> ch/dc a 500 W
e_total_bat_bike = 6.3 # Energia total das baterias das bicicletas (kWh)
# 0.63 kWh por bateria, sendo 12 V, são de 52.5 Ah
# BATERIA ESTACIONÁRIA
soc_max_est = 1 # soc Maximo da bateria
soc_min_est = 0.2 # soc minimo da bateria
soc_ini_est = 0.5
p_max_bat_est = 3 # Deve ser o ch e dc constante
e_total_bat_est = 2.4 # Energia total da bateria estacionária (kWh)
soc_ref_est = 0.5
# INVERSORES
eff_conv_w = 0.7 # 0.96 Eficiência conversor wireless
eff_conv_ac =  1 # 0.96
# Maximum power of inverters
p_max_inv_ac = 20 # (kW)
q_max_inv_ac = 20 # (kvar)


# PENALIDADES
peso_soc_est = 0.01
peso_variacao_est = 0.01
peso_p_rede_imp = (1-peso_soc_est-peso_variacao_est)/2
peso_p_rede_exp = 1-peso_p_rede_imp


''' ------- VARIÁVEIS DO PROBLEMA ------- '''
# REDE 
p_rede_exp_ac = pl.LpVariable.dicts('p_rede_exp_ac', 
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede_exp,
                                    cat='Continuous')
p_rede_imp_ac = pl.LpVariable.dicts('p_rede_imp_ac',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede_imp,
                                    cat='Continuous')


# Parte DC da rede
p_rede_exp_dc = pl.LpVariable.dicts('p_rede_exp_dc',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede_exp,
                                    cat='Continuous')
p_rede_imp_dc = pl.LpVariable.dicts('p_rede_imp_dc',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede_imp,
                                    cat='Continuous')
# Flags for network import
flag_rede_imp_ac = pl.LpVariable.dicts('flag_rede_imp_ac',
                                       ((tempo) for tempo in dados_entrada.index),
                                       cat='Binary')
flag_rede_imp_dc = pl.LpVariable.dicts('flag_rede_imp_dc',
                                       ((tempo) for tempo in dados_entrada.index),
                                       cat='Binary')
# Flags for network export
flag_rede_exp_ac = pl.LpVariable.dicts('flag_rede_exp_ac',
                                       ((tempo) for tempo in dados_entrada.index),
                                       cat='Binary')
flag_rede_exp_dc = pl.LpVariable.dicts('flag_rede_exp_dc',
                                       ((tempo) for tempo in dados_entrada.index),
                                       cat='Binary')

# BATERIA BIKE:
p_ch_bike1 = pl.LpVariable.dicts('p_ch_bike1',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bat_bike,
                                 cat='Continuous')
p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bat_bike,
                                 cat='Continuous')

p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=(p_max_bat_bike/eff_conv_w),
                                 cat='Continuous')

p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bat_bike*eff_conv_w,
                                 cat='Continuous')
# Flags for bike battery
flag_ch_bat_bike = pl.LpVariable.dicts('flag_ch_bat_bike1',
                                        ((tempo) for tempo in dados_entrada.index),
                                        cat='Binary')
flag_dc_bat_bike = pl.LpVariable.dicts('flag_dc_bat_bike1',
                                        ((tempo) for tempo in dados_entrada.index),
                                        cat='Binary')

# State Of Charge
soc_bike = pl.LpVariable.dicts('soc_bike',
                               ((tempo) for tempo in dados_entrada.index),
                               lowBound=soc_min_bike, upBound=soc_max_bike,
                               cat='Continuous')
soc_min_otm_bike = pl.LpVariable('soc_min_otm_bike', lowBound=soc_min_bike, 
                                 upBound=soc_max_bike,
                                 cat='Continuous')

# BATERIA ESTACIONÁRIA:
p_ch_bat_est = pl.LpVariable.dicts('p_ch_bat_est',
                                   ((tempo) for tempo in dados_entrada.index),
                                   lowBound=0, upBound=p_max_bat_est,
                                   cat='Continuous')
p_dc_bat_est = pl.LpVariable.dicts('p_dc_bat_est',
                                   ((tempo) for tempo in dados_entrada.index),
                                   lowBound=0, upBound=p_max_bat_est,
                                   cat='Continuous')
flag_ch_bat_est = pl.LpVariable.dicts('flag_ch_bat_est',
                                      ((tempo) for tempo in dados_entrada.index),
                                      cat='Binary')
flag_dc_bat_est = pl.LpVariable.dicts('flag_dc_bat_est',
                                      ((tempo) for tempo in dados_entrada.index),
                                      cat='Binary')


soc_est = pl.LpVariable.dicts('soc_est',
                              ((tempo) for tempo in dados_entrada.index),
                              lowBound=soc_min_est, upBound=soc_max_est,
                              cat='Continuous')

# MODULOS PARA A FUNÇÃO OBJETIVO
mod_dif_p_rede_imp = pl.LpVariable.dicts('mod_dif_p_rede_imp',
                                          ((tempo) for tempo in dados_entrada.index),
                                          lowBound=0, upBound=p_max_rede_imp,
                                          cat='Continuous')

mod_dif_p_rede_exp = pl.LpVariable.dicts('mod_dif_p_rede_exp',
                                          ((tempo) for tempo in dados_entrada.index),
                                          lowBound=0, upBound=p_max_rede_exp,
                                          cat='Continuous')

mod_variacao_p_est = pl.LpVariable.dicts('mod_variacao_p_est',
                                          ((tempo) for tempo in dados_entrada.index),
                                          lowBound=0, upBound=p_max_bat_est,
                                          cat='Continuous')

mod_dif_soc_ref_est = pl.LpVariable.dicts('mod_dif_soc_ref_est',
                                          ((tempo) for tempo in dados_entrada.index),
                                          lowBound=0, upBound=1,
                                          cat='Continuous')


dif_p_rede_imp = pl.LpVariable.dicts('dif_p_rede_imp',
                                      ((tempo) for tempo in dados_entrada.index),
                                      lowBound=-p_max_rede_imp, upBound=p_max_rede_imp,
                                      cat='Continuous')

dif_p_rede_exp = pl.LpVariable.dicts('dif_p_rede_exp',
                                      ((tempo) for tempo in dados_entrada.index),
                                      lowBound=-p_max_rede_exp, upBound=p_max_rede_exp,
                                      cat='Continuous')

dif_p_est = pl.LpVariable.dicts('dif_p_est',
                                      ((tempo) for tempo in dados_entrada.index),
                                      lowBound=-p_max_bat_est, upBound=p_max_bat_est,
                                      cat='Continuous')

dif_soc_ref_est = pl.LpVariable.dicts('dif_soc_ref_est',
                                      ((tempo) for tempo in dados_entrada.index),
                                      lowBound=-1, upBound=1,
                                      cat='Continuous')




''' ------- FUNÇÃO OBJETIVO ------- '''
prob += pl.lpSum([mod_dif_p_rede_imp[k]*(1/p_max_rede_imp)*peso_p_rede_imp for k in dados_entrada.index]) 
+ pl.lpSum([mod_dif_p_rede_exp[k]*(1/p_max_rede_exp)*peso_p_rede_exp for k in dados_entrada.index]) 
+ pl.lpSum([mod_dif_soc_ref_est[k]*peso_soc_est for k in dados_entrada.index])
# + pl.lpSum([mod_variacao_p_est[k]*peso_variacao_est for k in dados_entrada.index])

# prob += pl.lpSum([((p_rede_imp_ac[k]*ts*dados_entrada.loc[k,'custo_energia_imp']/(p_max_rede_imp*ts*custo_max_energia) - 
#                     p_rede_exp_ac[k]*ts*dados_entrada.loc[k,'custo_energia_exp']/(p_max_rede_exp*ts*custo_max_energia))*peso_p_rede +
#                     mod_dif_soc_ref_est[k]*peso_soc_est -
#                     soc_bike[k]*peso_soc_bike)
#                   for k in dados_entrada.index])




''' ------- RESTRIÇÕES ------- '''
for k in dados_entrada.index:
    
    # BATERIA BIKE:
    prob += p_ch_bike1[k] == p_max_bat_bike * flag_ch_bat_bike[k] # If use "<=", can p=0 and flag=1
    prob += p_dc_bike1[k] == p_max_bat_bike * flag_dc_bat_bike[k]

    prob += p_ch_bike2[k] == p_max_bat_bike / eff_conv_w * flag_ch_bat_bike[k]
    prob += p_dc_bike2[k] == p_max_bat_bike * eff_conv_w * flag_dc_bat_bike[k]
    
    prob += flag_ch_bat_bike[k] + flag_dc_bat_bike[k] <= 1 # simultaneity
    
    
    # Pega o soc_min_otm_bike
    # if k > 0:
    #     if (soc_bike[k] <= soc_min_otm_bike[k-1]):
    #         prob += soc_min_otm_bike[k] == soc_bike[k]
    
    prob += soc_min_otm_bike <= soc_bike[k]

    # BATERIA ESTACIONÁRIA:
    prob += p_ch_bat_est[k] <= p_max_bat_est * flag_ch_bat_est[k]
    prob += p_dc_bat_est[k] <= p_max_bat_est * flag_dc_bat_est[k]             
    prob += flag_dc_bat_est[k] + flag_ch_bat_est[k] <= 1 # simultaneity

    # SOC
    # if k == 0:
    #     prob += soc_bike[k] == soc_ini_bike
    #     prob += soc_est[k] == soc_ini_est
    # else:
    #     if (soc_bike[k-1] != 1):
    #         prob += soc_bike[k] == soc_bike[k-1] + p_ch_bike1[k-1]*ts/e_total_bat_bike
    #     else:
    #         prob += soc_bike[k] == soc_bike[k-1]

    # SOC
    if k == 0:
        prob += soc_bike[k] == soc_ini_bike
        prob += soc_est[k] == soc_ini_est
    else:
        # if (soc_bike[k] <= 0.99):
        #     prob += flag_ch_bat_bike[k] == 1
        #     prob += flag_dc_bat_bike[k] == 0
        # else:
        #     prob += flag_ch_bat_bike[k] == 0
        #     prob += flag_dc_bat_bike[k] == 0
            
        prob += soc_bike[k] ==  soc_bike[k-1] + (p_ch_bike1[k-1] 
                                               - p_dc_bike1[k-1])*ts/e_total_bat_bike
        prob += soc_est[k] ==  soc_est[k-1] + (p_ch_bat_est[k-1] 
                                               - p_dc_bat_est[k-1])*ts/e_total_bat_est


    # REDE
    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
    prob += p_rede_imp_ac[k] <= p_max_inv_ac * flag_rede_imp_ac[k]
    prob += p_rede_exp_ac[k] <= p_max_inv_ac * flag_rede_exp_ac[k]
    prob += flag_rede_imp_ac[k] + flag_rede_exp_ac[k] <= 1 # simultaneity
    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
    prob += p_rede_imp_dc[k] <= p_max_inv_ac * flag_rede_imp_dc[k]
    prob += p_rede_exp_dc[k] <= p_max_inv_ac * flag_rede_exp_dc[k]
    prob += flag_rede_imp_dc[k] + flag_rede_exp_dc[k] <= 1 # simultaneity
    
    prob += flag_rede_imp_ac[k] == flag_rede_imp_dc[k]
    prob += flag_rede_exp_ac[k] == flag_rede_exp_dc[k]

    # INVERSORES
    # EFICIÊNICA Conversor AC/DC
    prob += p_rede_exp_ac[k] == eff_conv_ac * p_rede_exp_dc[k]
    prob += p_rede_imp_dc[k] == eff_conv_ac * p_rede_imp_ac[k]
    # EFICIÊNICA Conversor Wireless
    prob += p_ch_bike1[k] == eff_conv_w * p_ch_bike2[k]
    prob += p_dc_bike2[k] == eff_conv_w * p_dc_bike1[k]

    # BALANÇO DE POTÊNCIA NO BARRAMENTO DC (Não considera a bat estacionária)
    prob += p_dc_bike2[k] + dados_entrada.loc[k,'potencia_PV'] + p_rede_imp_dc[
        k] + p_dc_bat_est[k] == p_ch_bike2[k] + p_rede_exp_dc[k] + p_ch_bat_est[k]
    
    # MÓDULOS PARA A FUNÇÃO OBJETIVO
    # Calcula o módulo da distância do soc_est de sua referência
    prob += dif_soc_ref_est[k] == soc_ref_est - soc_est[k]
    if (dif_soc_ref_est[k] >= 0):
        prob += mod_dif_soc_ref_est[k] == dif_soc_ref_est[k]
    else:
        prob += mod_dif_soc_ref_est[k] == -dif_soc_ref_est[k]
    
    # Calcula o códulo da diferença de pot da estacionária
    # prob += dif_p_est[k] == soc_ref_est - soc_est[k]
    # if (dif_soc_ref_est[k] >= 0):
    #     prob += mod_dif_soc_ref_est[k] == dif_soc_ref_est[k]
    # else:
    #     prob += mod_dif_soc_ref_est[k] == -dif_soc_ref_est[k]
      
    # Calcula o módulo da distância entre a referencia e a atual da potencia de importação
    prob += dif_p_rede_imp[k] == dados_ref.loc[k,'p_rede_imp_ref'] - p_rede_imp_ac[k]
    if (dif_p_rede_imp[k] >= 0):
        prob += mod_dif_p_rede_imp[k] == dif_p_rede_imp[k]
    else:
        prob += mod_dif_p_rede_imp[k] == -dif_p_rede_imp[k]
        
    # Calcula o módulo da distância entre a referencia e atual da potencia de exportação
    prob += dif_p_rede_exp[k] == dados_ref.loc[k,'p_rede_exp_ref'] - p_rede_exp_ac[k]
    if (dif_p_rede_exp[k] >= 0):
        prob += mod_dif_p_rede_exp[k] == dif_p_rede_exp[k]
    else:
        prob += mod_dif_p_rede_exp[k] == -dif_p_rede_exp[k]


''' ------- EXECUTA O ALGORITMO DE OTIMIZAÇÃO ------- '''
prob.solve(solver)



''' ------- VETORES PARA SALVAR RESULTADOS ------- '''
p_rede_ref = [0.0] * num_amostras
p_rede_res = [0.0] * num_amostras
p_rede_imp_res = [0.0] * num_amostras
p_rede_exp_res = [0.0] * num_amostras
somatorio_p_rede = [0.0] * num_amostras
p_ch_bike1_res = [0.0] * num_amostras
p_dc_bike1_res = [0.0] * num_amostras
p_ch_bat_est_res = [0.0] * num_amostras
p_dc_bat_est_res = [0.0] * num_amostras
p_pv_res = [0.0] * num_amostras
soc_bike_res = [0.0] * num_amostras
ch_dc_bike = [0.0] * num_amostras
soc_est_res = [0.0] * num_amostras
custo_energia_imp_res = [0.0] * num_amostras

mod_est = [0.0] * num_amostras

flag_rede_imp_ac_res = [0.0] * num_amostras
flag_rede_exp_ac_res = [0.0] * num_amostras
flag_ch_bat_bike_res = [0.0] * num_amostras
flag_dc_bat_bike_res = [0.0] * num_amostras




''' ------- IMPRIMIR DADOS ------- '''
for k in dados_entrada.index:
    p_rede_ref[k] = dados_ref.loc[k,'p_rede_ref']
    p_rede_imp_res[k] = p_rede_imp_ac[k].varValue
    p_rede_exp_res[k] = p_rede_exp_ac[k].varValue
    p_rede_res[k] = p_rede_exp_res[k] - p_rede_imp_res[k]
    p_ch_bike1_res[k] = p_ch_bike1[k].varValue
    p_dc_bike1_res[k] = p_dc_bike1[k].varValue
    ch_dc_bike[k] = p_ch_bike1_res[k] - p_dc_bike1_res[k]
    p_ch_bat_est_res[k] = p_ch_bat_est[k].varValue
    p_dc_bat_est_res[k] = p_dc_bat_est[k].varValue
    p_pv_res[k] = dados_entrada.loc[k,'potencia_PV']
    custo_energia_imp_res[k] = dados_entrada.loc[k,'custo_energia_exp']
    soc_bike_res[k] = soc_bike[k].varValue
    soc_est_res[k] = soc_est[k].varValue
    mod_est[k] = mod_dif_soc_ref_est[k].varValue
    
    flag_rede_imp_ac_res[k] = flag_rede_imp_ac[k].varValue
    flag_rede_exp_ac_res[k] = flag_rede_exp_ac[k].varValue
    flag_ch_bat_bike_res[k] = flag_ch_bat_bike[k].varValue
    flag_dc_bat_bike_res[k] = flag_dc_bat_bike[k].varValue


    if k >= 1:
        somatorio_p_rede[k] = somatorio_p_rede[k-1] + p_rede_res[k]*ts

# Inicial
somatorio_p_rede[0] = 0




""" ======= PLOTAR DADOS em 2 Gráficos ======= """

''' ---------------- GRÁFICO 1 --------------- '''
num_graf = 0
fig1,axs1 = plt.subplots(3)
# fig1.suptitle('Penalidades: Bike {peso_soc_bike}, Est.: {peso_soc_est}, p_rede = {peso_p_rede} \n Efic. conv. wireless = {ef}'.
#              format(peso_soc_bike=peso_soc_bike,peso_soc_est=peso_soc_est,peso_p_rede=peso_p_rede,ef=eff_conv_w),fontsize=10)
# Primeiro subplot
axs1[num_graf].step(time_array/4, p_rede_ref,c='r',linestyle='dashed',label='p_rede_ref [kW]')
axs1[num_graf].step(time_array/4, p_rede_res,c='b',label='p_rede_res [kW]')
# axs1[num_graf].step(time_array/4, p_rede_imp_res,c='g',label='p_rede_imp [kW]')
# axs1[num_graf].step(time_array/4, p_rede_exp_res,c='b',label='p_rede_exp [kW]')
# axs1[num_graf].step(time_array/4, p_pv_res,c='C1',linestyle='dashed',label='p_pv [kW]')
axs1[num_graf].legend(loc='lower center',prop={'size': 7})
# axs1[0].set_xticks(time_array)
axs1[num_graf].grid()
axs1[num_graf].set_ylabel('Amplitude')
axs1[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs1[num_graf].tick_params(axis='y', which='major', labelsize=8)
axs1[num_graf].set_yticks([min(p_rede_res),max(p_rede_res)/2,max(p_rede_res)])
axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])
num_graf +=1
# Segundo subplot
axs1[num_graf].step(time_array/4, soc_bike_res,c='g',label='soc_bike')
# axs1[num_graf].step(time_array/4, ch_dc_bike,c='#1f77b4',label='ch_dc_bike')
axs1[num_graf].step(time_array/4, soc_est_res,c='#1f77b4',label='soc_est')
axs1[num_graf].step(time_array/4, custo_energia_imp_res,c='C1',label='custo_energia_imp_res [R$/(kWh)]')
axs1[num_graf].legend(loc='lower center',prop={'size': 7})
# axs1[1].set_xticks(time_array)
axs1[num_graf].grid()
axs1[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs1[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs1[num_graf].set_yticks([0,0.2,0.5,1])
axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])

num_graf +=1
# Terceiro subplot
axs1[num_graf].step(time_array/4, somatorio_p_rede,c='#d62728',label='somatorio_p_rede [kWh]')
axs1[num_graf].legend(loc='lower right',prop={'size': 7})
axs1[num_graf].tick_params(axis='y', which='major', labelsize=8)
axs1[num_graf].grid()
axs1[num_graf].set_xlabel('Tempo (horas)')
axs1[num_graf].set_yticks([min(somatorio_p_rede)*2,0,max(somatorio_p_rede)/2,max(somatorio_p_rede)])
axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])



# ''' ---------------- GRÁFICO 2 --------------- '''
''' Bateria Bike [kWh] '''
num_graf = 0
fig2,axs2 = plt.subplots(3)
axs2[num_graf].step(time_array/4, p_ch_bike1_res,c='g',label='p_ch_bike1_res [kW]')
axs2[num_graf].step(time_array/4, p_dc_bike1_res,c='r',label='p_dc_bike1_res [kW]')
axs2[num_graf].legend(loc='lower center',prop={'size': 7})

# axs1[0].set_xticks(time_array)
axs2[num_graf].grid()
axs2[num_graf].set_ylabel('Amplitude')
axs2[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs2[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs2[num_graf].set_yticks([-0.7,-0.5,0,0.5,0.7])
axs2[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])

''' Flags '''
num_graf +=1
axs2[num_graf].step(time_array/4, flag_rede_imp_ac_res,c='r',linestyle='dashed',label='flag_rede_imp_ac_res')
axs2[num_graf].step(time_array/4, flag_rede_exp_ac_res,c='g',label='flag_rede_exp_ac_res')
axs2[num_graf].legend(loc='lower center',prop={'size': 7})

axs2[num_graf].grid()
axs2[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs2[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs2[num_graf].set_yticks([0,1])
axs2[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])


num_graf +=1
axs2[num_graf].step(time_array/4, p_ch_bat_est_res,c='b',linestyle='dashed',label='p_ch_bat_est_res')
axs2[num_graf].step(time_array/4, p_dc_bat_est_res,c='C1',label='p_dc_bat_est_res')
axs2[num_graf].legend(loc='lower right',prop={'size': 7})

axs2[num_graf].grid()
axs2[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs2[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs2[num_graf].set_yticks([0,p_max_bat_est])
axs2[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])




# name_figure = "220810.png"
# plt.savefig(name_figure, format="png", dpi=400)
# plt.show()




''' WRITE RESULTS IN SCV'''
# p_rede_df = pd.DataFrame(p_rede_res)
# p_rede_df.to_csv("p_rede_ref.csv", index=False)

# result = {
#     'p_rede_ref' : p_rede_res,
#     'p_rede_imp_ref' : p_rede_imp_res,
#     'p_rede_exp_ref' : p_rede_exp_res
#     }

# df = pd.DataFrame(result)
# df.to_csv("p_rede_ref.csv", index=False,sep=';')