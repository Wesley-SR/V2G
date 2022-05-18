'''============================================================================
PROJETO V2G
MODELO DE OTIMIZAÇÃO 1 - MILP

Mudança em relação ao código anterior:
#==========================================================================='''

import pulp as pl
import pandas as pd
import matplotlib.pyplot as plt

''' ------- DEFINIÇÃO DO PROBLEMA ------- '''
dados_entrada = pd.read_csv('dados_entrada.csv', index_col=['tempo'])
prob = pl.LpProblem("otimizacao_V2G", pl.LpMinimize)
solver = pl.PULP_CBC_CMD()
# Solver Disponívels
# ['PULP_CBC_CMD', 'MIPCL_CMD', 'SCIP_CMD']
# Solver Possíveis
# ['GLPK_CMD', 'PYGLPK', 'CPLEX_CMD', 'CPLEX_PY', 'CPLEX_DLL', 'GUROBI', 
# 'GUROBI_CMD', 'MOSEK', 'XPRESS', 'PULP_CBC_CMD', 'COIN_CMD', 'COINMP_DLL', 
# 'CHOCO_CMD', 'MIPCL_CMD', 'SCIP_CMD']


''' ------- PARÂMETROS ------- ''' 
p_max_rede = 20
p_max_pv = 10
soc_max = 1 # soc Maximo da bateria
soc_min = 0.2 # soc minimo da bateria
soc_ini_bike = 0.2
soc_ini_est = 0.5
ts = 1 # Intervalo de tempo (0.16667h = 10 min)
p_max_bike = 0.5
p_max_bat_est = 3
E_bat_est = 2.4
E_bat_bike = 2.4 # Energia total das baterias das bicicletas (kWh)
# Efficiency of inverters
eff_conv_w = 1 # 0.96
eff_conv_ac = 1 # 0.96
# Maximum power of inverters
p_max_inv_ac = 20 # (kW)
q_max_inv_ac = 20 # (kvar)
lambda_e_max = 1

peso_soc_bike = 0.5 # Valor que multiplica o soc_bike para gerar a penalidade
peso_p_rede = 1 - peso_soc_bike

''' ------- VARIÁVEIS DO PROBLEMA ------- '''
# Parte AC na rede
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

# Bike battery charging
p_ch_bike1 = pl.LpVariable.dicts('p_ch_bike1',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bike,
                                 cat='Continuous')
p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bike,
                                 cat='Continuous')

# Bike battery discharge
p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bike,
                                 cat='Continuous')
p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2',
                                 ((tempo) for tempo in dados_entrada.index),
                                 lowBound=0, upBound=p_max_bike,
                                 cat='Continuous')

# Stationary battery charging
p_ch_bat_est = pl.LpVariable.dicts('p_ch_bat_est',
                                   ((tempo) for tempo in dados_entrada.index),
                                   lowBound=0, upBound=p_max_bat_est,
                                   cat='Continuous')

# Stationary battery discharge
# p_dc_bat_est = pl.LpVariable.dicts('p_dc_bat_est',
#                                     ((tempo) for tempo in dados_entrada.index),
#                                     lowBound=0, upBound=p_max_bat_est,
#                                     cat='Continuous')
# Flags for charg and discharg
# flag_ch_bat_est = pl.LpVariable.dicts('flag_ch_bat_est',
#                                       ((tempo) for tempo in dados_entrada.index),
#                                       cat='Binary')
# flag_dc_bat_est = pl.LpVariable.dicts('flag_dc_bat_est',
#                                       ((tempo) for tempo in dados_entrada.index),
#                                       cat='Binary')

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

# State Of Charge
soc_bike = pl.LpVariable.dicts('soc_bike',
                               ((tempo) for tempo in dados_entrada.index),
                               lowBound=soc_min, upBound=soc_max,
                               cat='Continuous')
# soc_est = pl.LpVariable.dicts('soc_est',
#                               ((tempo) for tempo in dados_entrada.index),
#                               lowBound=soc_min, upBound=soc_max,
#                               cat='Continuous')

# Penalty for force change battery
penalty = pl.LpVariable.dict('penalty',
                             ((tempo) for tempo in dados_entrada.index),
                             lowBound=0, upBound=100, cat='Continuous')



''' ------- RESTRIÇÕES ------- '''
for i in dados_entrada.index:
    
    # BATERIA BIKE:
    prob += p_ch_bike1[i] == p_max_bike * flag_ch_bat_bike1[i]
    prob += p_dc_bike1[i] <= p_max_bike * flag_dc_bat_bike1[i]
    prob += flag_ch_bat_bike1[i] + flag_dc_bat_bike1[i] <= 1 # simultaneity
    
    prob += p_ch_bike2[i] == p_max_bike * flag_ch_bat_bike2[i]
    prob += p_dc_bike2[i] <= p_max_bike * flag_dc_bat_bike2[i]
    prob += flag_ch_bat_bike2[i] + flag_dc_bat_bike2[i] <= 1 # simultaneity
    
    # BATERIA ESTACIONÁRIA:
    # prob += p_ch_bat_est[i] <= p_max_bat_est * flag_ch_bat_est[i]
    # prob += p_dc_bat_est[i] <= p_max_bat_est * flag_dc_bat_est[i]             
    # prob += flag_dc_bat_est[i] + flag_ch_bat_est[i] <= 1 # simultaneity

    # p_rede
    prob += p_rede[i] == p_rede_imp_ac[i] - p_rede_exp_ac[i]

    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
    prob += p_rede_imp_ac[i] <= p_max_inv_ac * flag_rede_imp_ac[i]
    prob += p_rede_exp_ac[i] <= p_max_inv_ac * flag_rede_exp_ac[i]
    prob += flag_rede_imp_ac[i] + flag_rede_exp_ac[i] <= 1 # simultaneity

    # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
    prob += p_rede_imp_dc[i] <= p_max_inv_ac * flag_rede_imp_dc[i]
    prob += p_rede_exp_dc[i] <= p_max_inv_ac * flag_rede_exp_dc[i]
    prob += flag_rede_imp_dc[i] + flag_rede_exp_dc[i] <= 1 # simultaneity

    # EFICIÊNICA Conversor AC/DC
    prob += p_rede_exp_ac[i] == eff_conv_ac*p_rede_exp_dc[i]
    prob += p_rede_imp_dc[i] == eff_conv_ac*p_rede_imp_ac[i]

    # EFICIÊNICA Conversor Wireless
    prob += p_ch_bike1[i] == eff_conv_w*p_ch_bike2[i]
    prob += p_dc_bike2[i] == eff_conv_w*p_dc_bike1[i]

    # BALANÇO DE POTÊNCIA NO BARRAMENTO DC (Não considera a bat estacionária)
    prob += p_dc_bike2[i] + dados_entrada.loc[i,'potencia_PV'] + p_rede_imp_dc[
        i] == p_ch_bike2[i] + p_rede_exp_dc[i]

    # Calculo do Estado de Carga (soc) - Baterias Bike
    if i == 0:
        prob += soc_bike[i] == soc_ini_bike
        # prob += soc_est[i] == soc_ini_est
    else:
        prob += soc_bike[i] ==  soc_bike[i-1] + (p_ch_bike1[i-1] -
                                                 p_dc_bike1[i-1])*ts/E_bat_bike
        # prob += soc_est[i] ==  soc_est[i-1] + (p_ch_bat_est[i-1] 
        #                                        - p_dc_bat_est[i-1]) *
        # ts / E_bat_est
    
    # prob += penalty[i] == soc_bike[i]



''' ------- FUNÇÃO OBJETIVO ------- '''
# Somatório das 24 horas da potência*tempo*custo_energia
prob += pl.lpSum([((p_rede[i]*ts*dados_entrada.loc[i,'custo_energia'])/
                   (p_max_rede*ts*lambda_e_max)*peso_p_rede -
                   soc_bike[i]*peso_soc_bike)
                  for i in dados_entrada.index])



''' ------- EXECUTA O ALGORITMO DE OTIMIZAÇÃO ------- '''
prob.solve(solver)


''' ------- VETORES PARA SALVAR RESULTADOS ------- '''
p_rede_res = [0.0] * 25
p_ch_bike1_res = [0.0] * 25
p_dc_bike1_res = [0.0] * 25
p_ch_bat_est_res = [0.0] * 25
p_dc_bat_est_res = [0.0] * 25
p_pv_res = [0.0] * 25
soc_bike_res = [0.0] * 25
ch_dc_bike = [0.0] * 25
custo_energia = [0.0] * 25
tempo_res = dados_entrada.index

''' ------- IMPRIMIR DADOS ------- '''
for i in dados_entrada.index:
        p_rede_res[i] = p_rede[i].varValue
        print('p_rede_res[{i}] = {Value}'.format(i=i,Value=p_rede_res[i]))
        
        p_ch_bike1_res[i] = p_ch_bike1[i].varValue
        print('p_ch_bike1_res[{i}] = {Value}'.format(
            i=i,Value=p_ch_bike1_res[i]))

        p_dc_bike1_res[i] = p_dc_bike1[i].varValue
        print('p_dc_bike1_res[{i}] = {Value}'.format(
            i=i,Value=p_dc_bike1_res[i]))

        ch_dc_bike[i] = p_ch_bike1_res[i] - p_dc_bike1_res[i]

        p_pv_res[i] = dados_entrada.loc[i,'potencia_PV']
        print('p_pv_res[{i}] = {Value}'.format(i=i,Value=p_pv_res[i]))

        custo_energia[i] = dados_entrada.loc[i,'custo_energia']
        print('custo_energia[{i}] = {Value}'.format(i=i,Value=custo_energia[i]))

        soc_bike_res[i] = soc_bike[i].varValue
        print('soc_bike_res[{i}] = {Value}\n\n'.format(
            i=i,Value=soc_bike_res[i]))

        # p_ch_bat_est_res[i] = p_ch_bat_est[i].varValue
        # print('p_ch_bat_est_res[{i}] = {Value}'.format(
        #     i=i,Value=p_ch_bat_est_res[i]))
        
        # p_dc_bat_est_res[i] = p_dc_bat_est[i].varValue
        # print('p_dc_bat_est_res[{i}] = {Value}\n\n'.format(
        #     i=i,Value=p_dc_bat_est_res[i]))


''' ------- PLOTAR DADOS em 1 Gráfico ------- '''
# CONTÍNUO
# fig,ax1 = plt.subplots()

# ax1.plot(tempo_res,p_rede_res,'b',label='p_rede_imp')
# ax1.plot(tempo_res,p_pv_res,'r',label='p_pv')
# ax1.plot(tempo_res,soc_bike_res,'g',label='soc_bike')
# ax1.plot(tempo_res,p_ch_bike1_res,'k',label='p_ch_bike1_res*10')

# # DISCRETO
# # plt.stem(tempo_res,p_rede_res,'--b',markerfmt='bo',
# #                        label='p_rede_imp')
# # plt.stem(tempo_res,p_pv_res,'--r',markerfmt='ro',
# #                        label='p_pv')
# # plt.stem(tempo_res,soc_bike_res,'--g',markerfmt='go',
# #                        label='soc_bike*10')

# ax1.set_xlabel('Tempo (horas)')
# ax1.set_ylabel('Amplitude')
# # ax1.set_title('Penalidade = soc_bike[i]*{peso_soc_bike}'.format(peso_soc_bike=peso_soc_bike))
# ax1.set_legend(loc="lower left")
# ax1.set_xticks(tempo_res)


''' ------- PLOTAR DADOS em 2 Gráficos ------- '''
fig,axs = plt.subplots(2)
fig.suptitle('Penalidade = soc_bike[i]*{peso_soc_bike}'.format(peso_soc_bike=peso_soc_bike))
axs[0].step(tempo_res, p_rede_res,c='#d62728',label='p_rede')
axs[0].step(tempo_res, p_pv_res,c='c',label='p_pv')
axs[0].legend(loc='upper right')
axs[0].set_xticks(tempo_res)
axs[0].grid()
axs[0].set_ylabel('Amplitude')

axs[1].step(tempo_res, soc_bike_res,c='g',label='soc_bike')
axs[1].step(tempo_res, ch_dc_bike,c='#1f77b4',label='ch_dc_bike')
axs[1].step(tempo_res, custo_energia,c='C1',label='custo_energia')
axs[1].legend(loc='lower center')
axs[1].set_xticks(tempo_res)
axs[1].grid()
axs[1].set_xlabel('Tempo (horas)')
axs[1].set_ylabel('Amplitude')

name_figure = "Imagens_ts1h/penalidade_soc_{}.png".format(peso_soc_bike)
plt.savefig(name_figure, format="png", dpi=400)
plt.show()