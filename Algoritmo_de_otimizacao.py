'''============================================================================
PROJETO V2G
MODELO DE OTIMIZAÇÃO 1 - MILP

Mudança em relação ao código anterior:
    - Os pesos voltaram a ser fixo no tempo
    - Retirada de bicicletas alterando a energia total
Observações:
#==========================================================================='''


import pulp as pl
import pandas as pd
import matplotlib.pyplot as plt

''' ------- DEFINIÇÃO DO PROBLEMA ------- '''
dados_entrada = pd.read_csv('dados_entrada_retirando_bikes.csv', index_col=['tempo'])
prob = pl.LpProblem("otimizacao_V2G", pl.LpMinimize)

solver = pl.PULP_CBC_CMD() # pl.SCIP_CMD()
# Solver Disponívels
# ['PULP_CBC_CMD', 'MIPCL_CMD', 'SCIP_CMD']
# Solver Possíveis
# ['GLPK_CMD', 'PYGLPK', 'CPLEX_CMD', 'CPLEX_PY', 'CPLEX_DLL', 'GUROBI', 
# 'GUROBI_CMD', 'MOSEK', 'XPRESS', 'PULP_CBC_CMD', 'COIN_CMD', 'COINMP_DLL', 
# 'CHOCO_CMD', 'MIPCL_CMD', 'SCIP_CMD']

time_array = dados_entrada.index

''' ------- PARÂMETROS ------- ''' 
ts = 0.16667 # Intervalo de tempo (0.16667h = 10 min)
num_amostras = time_array.size
# REDE
p_max_rede = 20
p_max_pv = 10
custo_max_energia = 1
# BATERIA BICICLETA
num_bikes = 10
soc_max_bike = 1 # soc Maximo da bateria
soc_min_bike = 0 # soc minimo da bateria
soc_ini_bike = 0.5
p_max_bat_bike = 0.5 # (kW) -> ch/dc a 500 W
e_total_bat_bike = 6.3 # 6.3 # Energia total das baterias das bicicletas (kWh)
# 0.63 kWh por bateria, sendo 12 V, são de 52.5 Ah por bateria
# BATERIA ESTACIONÁRIA
soc_max_est = 1 # soc Maximo da bateria
soc_min_est = 0.2 # soc minimo da bateria
soc_ini_est = 0.6
p_max_bat_est = 3 # Deve ser o ch e dc constante (kW)
e_total_bat_est = 9.6 # Energia total da bateria estacionária (kWh)
soc_ref_est = 0.5
# INVERSORES
eff_conv_w = 0.6 # Eficiência conversor wireless
eff_conv_ac =  0.99 # 0.96
# Maximum power of inverters
p_max_inv_ac = 20 # (kW)
q_max_inv_ac = 20 # (kvar)

# PENALIDADES -> Ajustes de acordo com a experiência do projetista e a demanda
# das bicicletas
# peso_soc_bike + peso_soc_est + peso_p_rede = 1

peso_soc_bike = 0.05
peso_soc_es = 0.01
peso_p_rede = 1 - peso_soc_bike - peso_soc_es


# soc_bike = [num_bikes][num_amostras]


''' ------- VARIÁVEIS DO PROBLEMA ------- '''
# REDE 
p_rede_exp_ac = pl.LpVariable.dicts('p_rede_exp_ac', 
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede,
                                    cat='Continuous')
p_rede_imp_ac = pl.LpVariable.dicts('p_rede_imp_ac',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede,
                                    cat='Continuous')
p_rede = pl.LpVariable.dicts('p_rede',
                             ((tempo) for tempo in dados_entrada.index),
                             lowBound=-p_max_rede, upBound=p_max_rede,
                             cat='Continuous')
# q_rede_exp_ac = pl.LpVariable.dicts('q_rede_exp_ac',
#                                     ((tempo) for tempo in dados_entrada.index),
#                                     lowBound=0, upBound=p_max_rede,
#                                     cat='Continuous')
# q_rede_imp_ac = pl.LpVariable.dicts('q_rede_imp_ac',
#                                     ((tempo) for tempo in dados_entrada.index),
#                                     lowBound=0, upBound=p_max_rede,
#                                     cat='Continuous')
# Parte DC da rede
p_rede_exp_dc = pl.LpVariable.dicts('p_rede_exp_dc',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede,
                                    cat='Continuous')
p_rede_imp_dc = pl.LpVariable.dicts('p_rede_imp_dc',
                                    ((tempo) for tempo in dados_entrada.index),
                                    lowBound=0, upBound=p_max_rede,
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
p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=(p_max_bat_bike/eff_conv_w),
                                 cat='Continuous')
p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bat_bike,
                                 cat='Continuous')
p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bat_bike,
                                 cat='Continuous')
# Flags for bike battery
flag_ch_bat_bike1 = pl.LpVariable.dicts('flag_ch_bat_bike1',
                                        ((tempo) for tempo in dados_entrada.index),
                                        cat='Binary')
flag_dc_bat_bike1 = pl.LpVariable.dicts('flag_dc_bat_bike1',
                                        ((tempo) for tempo in dados_entrada.index),
                                        cat='Binary')
flag_ch_bat_bike2 = pl.LpVariable.dicts('flag_ch_bat_bike2',
                                        ((tempo) for tempo in dados_entrada.index),
                                        cat='Binary')
flag_dc_bat_bike2 = pl.LpVariable.dicts('flag_dc_bat_bike2',
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
dif_soc_ref_est = pl.LpVariable.dicts('dif_soc_ref_est',
                                      ((tempo) for tempo in dados_entrada.index),
                                      lowBound=-1, upBound=1,
                                      cat='Continuous')
mod_dif_soc_ref_est = pl.LpVariable.dicts('mod_dif_soc_ref_est',
                                          ((tempo) for tempo in dados_entrada.index),
                                          lowBound=0, upBound=1,
                                          cat='Continuous')
soc_est = pl.LpVariable.dicts('soc_est',
                              ((tempo) for tempo in dados_entrada.index),
                              lowBound=soc_min_est, upBound=soc_max_est,
                              cat='Continuous')



''' ------- RESTRIÇÕES ------- '''
for k in dados_entrada.index:
    
    # BATERIA BIKE:
    prob += p_ch_bike1[k] == p_max_bat_bike * flag_ch_bat_bike1[k]
    prob += p_dc_bike1[k] <= p_max_bat_bike * flag_dc_bat_bike1[k]
    prob += flag_ch_bat_bike1[k] + flag_dc_bat_bike1[k] <= 1 # simultaneity
    prob += p_ch_bike2[k] <= p_max_bat_bike/eff_conv_w * flag_ch_bat_bike2[k]
    prob += p_dc_bike2[k] <= p_max_bat_bike * flag_dc_bat_bike2[k]
    prob += flag_ch_bat_bike2[k] + flag_dc_bat_bike2[k] <= 1 # simultaneity
        
    # Pega o soc_min_otm_bike
    # if k > 0:
    #     if (soc_bike[k] <= soc_min_otm_bike[k-1]):
    #         prob += soc_min_otm_bike[k] == soc_bike[k]
    
    prob += soc_min_otm_bike <= soc_bike[k]

    # BATERIA ESTACIONÁRIA:
    prob += p_ch_bat_est[k] <= p_max_bat_est * flag_ch_bat_est[k]
    prob += p_dc_bat_est[k] <= p_max_bat_est * flag_dc_bat_est[k]             
    prob += flag_dc_bat_est[k] + flag_ch_bat_est[k] <= 1 # simultaneity
    # Calcula o módulo da distância do soc_est de sua referência
    prob += dif_soc_ref_est[k] == soc_ref_est - soc_est[k]
    if (dif_soc_ref_est[k] >= 0):
        prob += mod_dif_soc_ref_est[k] == dif_soc_ref_est[k]
    else:
        prob += mod_dif_soc_ref_est[k] == -dif_soc_ref_est[k]

    # SOC
    if k == 0:
        prob += soc_bike[k] == soc_ini_bike
        prob += soc_est[k] == soc_ini_est
    else:
        prob += soc_bike[k] ==  soc_bike[k-1] + (p_ch_bike1[k-1] -
                                                 p_dc_bike1[k-1])*ts/e_total_bat_bike
        prob += soc_est[k] ==  soc_est[k-1] + (p_ch_bat_est[k-1] 
                                                - p_dc_bat_est[k-1])*ts/e_total_bat_est

    # REDE
    prob += p_rede[k] == (p_rede_imp_ac[k] - p_rede_exp_ac[k])
    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
    prob += p_rede_imp_ac[k] <= p_max_inv_ac * flag_rede_imp_ac[k]
    prob += p_rede_exp_ac[k] <= p_max_inv_ac * flag_rede_exp_ac[k]
    prob += flag_rede_imp_ac[k] + flag_rede_exp_ac[k] <= 1 # simultaneity
    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
    prob += p_rede_imp_dc[k] <= p_max_inv_ac * flag_rede_imp_dc[k]
    prob += p_rede_exp_dc[k] <= p_max_inv_ac * flag_rede_exp_dc[k]
    prob += flag_rede_imp_dc[k] + flag_rede_exp_dc[k] <= 1 # simultaneity

    # INVERSORES
    # EFICIÊNICA Conversor AC/DC
    prob += p_rede_exp_ac[k] == eff_conv_ac * p_rede_exp_dc[k]
    prob += p_rede_imp_dc[k] == eff_conv_ac * p_rede_imp_ac[k]
    # EFICIÊNICA Conversor Wireless
    prob += p_ch_bike1[k] == eff_conv_w * p_ch_bike2[k]
    prob += p_dc_bike2[k] == eff_conv_w * p_dc_bike1[k]

    # BALANÇO DE POTÊNCIA NO BARRAMENTO DC
    prob += p_dc_bike2[k] + dados_entrada.loc[k,'potencia_PV'] + p_rede_imp_dc[
        k] + p_dc_bat_est[k] == p_ch_bike2[k] + p_rede_exp_dc[k] + p_ch_bat_est[k]




''' ------- FUNÇÃO OBJETIVO ------- '''
# Somatório das 24 horas da potência*tempo*custo_energia
prob += pl.lpSum([((p_rede[k]*ts*dados_entrada.loc[k,'custo_energia'])/
                  (p_max_rede*ts*custo_max_energia)*peso_p_rede +
                  mod_dif_soc_ref_est[k]*peso_soc_es -
                  soc_bike[k]*peso_soc_bike)
                  for k in dados_entrada.index])

# soc_min_otm_bike*peso_soc_bike


''' ------- EXECUTA O ALGORITMO DE OTIMIZAÇÃO ------- '''
prob.solve(solver)



''' ------- VETORES PARA SALVAR RESULTADOS ------- '''
p_rede_res = [0.0] * num_amostras
somatorio_p_rede = [0.0] * num_amostras
p_ch_bike1_res = [0.0] * num_amostras
p_dc_bike1_res = [0.0] * num_amostras
p_ch_bat_est_res = [0.0] * num_amostras
p_dc_bat_est_res = [0.0] * num_amostras
p_pv_res = [0.0] * num_amostras
soc_bike_res = [0.0] * num_amostras
ch_dc_bike = [0.0] * num_amostras
soc_est_res = [0.0] * num_amostras
custo_energia = [0.0] * num_amostras
mod_est = [0.0] * num_amostras



''' ------- IMPRIMIR DADOS ------- '''
for k in dados_entrada.index:
    p_rede_res[k] = - p_rede[k].varValue
    p_ch_bike1_res[k] = p_ch_bike1[k].varValue
    p_dc_bike1_res[k] = p_dc_bike1[k].varValue
    ch_dc_bike[k] = p_ch_bike1_res[k] - p_dc_bike1_res[k]
    p_pv_res[k] = dados_entrada.loc[k,'potencia_PV']
    custo_energia[k] = dados_entrada.loc[k,'custo_energia']
    soc_bike_res[k] = soc_bike[k].varValue
    soc_est_res[k] = soc_est[k].varValue
    mod_est[k] = mod_dif_soc_ref_est[k].varValue
    
    
    if k >= 1:
        somatorio_p_rede[k] = somatorio_p_rede[k-1] + p_rede_res[k]

# Inicial
somatorio_p_rede[0] = p_rede_res[0]



''' ------- PLOTAR DADOS em 1 Gráfico ------- '''
# CONTÍNUO
# fig,ax1 = plt.subplots()

# ax1.plot(time_array,p_rede_res,'b',label='p_rede_imp')
# ax1.plot(time_array,p_pv_res,'r',label='p_pv')
# ax1.plot(time_array,soc_bike_res,'g',label='soc_bike')
# ax1.plot(time_array,p_ch_bike1_res,'k',label='p_ch_bike1_res*10')

# # DISCRETO
# # plt.stem(time_array,p_rede_res,'--b',markerfmt='bo',
# #                        label='p_rede_imp')
# # plt.stem(time_array,p_pv_res,'--r',markerfmt='ro',
# #                        label='p_pv')
# # plt.stem(time_array,soc_bike_res,'--g',markerfmt='go',
# #                        label='soc_bike*10')

# ax1.set_xlabel('Tempo (horas)')
# ax1.set_ylabel('Amplitude')
# # ax1.set_title('Penalidade = soc_bike[k]*{peso_soc_bike}'.format(peso_soc_bike=peso_soc_bike))
# ax1.set_legend(loc="lower left")
# ax1.set_xticks(time_array)


''' ------- PLOTAR DADOS em 2 Gráficos ------- '''
num_graf = 0
fig,axs = plt.subplots(3)
# fig.suptitle('Penalidades: Bike {peso_soc_bike}, Est.: {peso_soc_est}, p_rede = {peso_p_rede} \n Efic. conv. wireless = {ef}'.
#              format(peso_soc_bike=peso_soc_bike,peso_soc_est=peso_soc_est,peso_p_rede=peso_p_rede,ef=eff_conv_w),fontsize=10)
# Primeiro subplot
axs[num_graf].step(time_array/6, p_rede_res,c='#d62728',label='p_rede [kW] inversa')
axs[num_graf].step(time_array/6, p_pv_res,c='c',label='p_pv [kW]')
axs[num_graf].legend(loc='upper right',prop={'size': 7})
# axs[0].set_xticks(time_array)
axs[num_graf].grid()
axs[num_graf].set_ylabel('Amplitude')
axs[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs[num_graf].set_yticks([-20,-10,0,10,20])
axs[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])
num_graf +=1
# Segundo subplot
axs[num_graf].step(time_array/6, soc_bike_res,c='g',label='soc_bike')
# axs[num_graf].step(time_array/6, ch_dc_bike,c='#1f77b4',label='ch_dc_bike')
axs[num_graf].step(time_array/6, soc_est_res,c='#1f77b4',label='soc_est')
axs[num_graf].step(time_array/6, custo_energia,c='C1',label='custo_energia [R$/(kWh)]')
axs[num_graf].legend(loc='lower center',prop={'size': 7})
# axs[1].set_xticks(time_array)
axs[num_graf].grid()
axs[num_graf].tick_params(axis='x', which='major', labelsize=7)
axs[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs[num_graf].set_yticks([0,0.2,0.5,1])
axs[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])

num_graf +=1
# Terceiro subplot
axs[num_graf].step(time_array/6, somatorio_p_rede,c='#d62728',label='somatorio_p_rede [kW]')
axs[num_graf].legend(loc='lower right',prop={'size': 7})
axs[num_graf].tick_params(axis='y', which='major', labelsize=10)
axs[num_graf].grid()
axs[num_graf].set_xlabel('Tempo (horas)')
axs[num_graf].set_yticks([-100,0,200,400,500])
axs[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])

# name_figure = "imagens_testes/220419_penalidade_soc_{}.png".format(peso_soc_bike)
# plt.savefig(name_figure, format="png", dpi=400)
plt.show()



