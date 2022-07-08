#!/usr/bin/env python3

import json
import pandas as pd
import pulp as pl

class ModelPredictiveControl:
    def __init__(self, config_file_path, path_to_inicial_data):

        # Load initial datas
        with open(config_file_path, 'r') as json_file:
            self.config_data = json.load(json_file)
        self.inicial_data = pd.read_csv(path_to_inicial_data, index_col=['self.time'])
        
        # Optimization problem
        self.prob = pl.LpProblem(self.config_data["optimization_name"], pl.LpMinimize)
        # Solver
        self.solver = pl.PULP_CBC_CMD()
        # Parameters
        self.sample_time = float(self.config_data["sample_time"])
        self.number_of_samples = self.inicial_data.index.size
        self.num_bikes = int(self.config_data["num_bikes"])
        self.bikes = range(0, self.num_bikes)
        self.time = range(0, self.number_of_samples+1)
        # self.soc_bike = range(0, self.num_bikes)

    
    def problem_variables(self):
        self.p_rede_exp_ac = pl.LpVariable.dicts('p_rede_exp_ac', range(0, self.number_of_samples),
                                            lowBound=0, upBound= float(self.config_data["p_max_rede"]),
                                            cat='Continuous')
        
        self.p_rede_imp_ac = pl.LpVariable.dicts('p_rede_imp_ac', range(0, self.number_of_samples),
                                            lowBound=0, upBound= float(self.config_data["p_max_rede"]),
                                            cat='Continuous')
        
        self.p_rede = pl.LpVariable.dicts('p_rede', range(0, self.number_of_samples),
                                     lowBound= - float(self.config_data["p_max_rede"]), upBound= float(self.config_data["p_max_rede"]),
                                     cat='Continuous')
        
        # q_rede_exp_ac = pl.LpVariable.dicts('q_rede_exp_ac', range(0,self.number_of_samples),
        #                                     lowBound=0, upBound=p_max_rede,
        #                                     cat='Continuous')
        # q_rede_imp_ac = pl.LpVariable.dicts('q_rede_imp_ac', range(0,self.number_of_samples),
        #                                     lowBound=0, upBound=p_max_rede,
        #                                     cat='Continuous')

        # Parte DC da rede
        self.p_rede_exp_dc = pl.LpVariable.dicts('p_rede_exp_dc', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.config_data["p_max_rede"]),
                                            cat='Continuous')
        self.p_rede_imp_dc = pl.LpVariable.dicts('p_rede_imp_dc', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.config_data["p_max_rede"]),
                                            cat='Continuous')
        # Flags for network import
        self.flag_rede_imp_ac = pl.LpVariable.dicts('flag_rede_imp_ac', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        self.flag_rede_imp_dc = pl.LpVariable.dicts('flag_rede_imp_dc', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        # Flags for network export
        self.flag_rede_exp_ac = pl.LpVariable.dicts('flag_rede_exp_ac', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        self.flag_rede_exp_dc = pl.LpVariable.dicts('flag_rede_exp_dc', range(0, int(self.number_of_samples)),
                                               cat='Binary')

        # BATERIA BIKE:
        self.p_ch_bike1 = pl.LpVariable.dicts('p_ch_bike1', (self.bikes, self.time),
                                         lowBound = 0,
                                         upBound = float(self.config_data["p_max_bat_bike"]),
                                         cat='Continuous')
        self.p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = (float(self.config_data["p_max_bat_bike"])/float(self.config_data["eff_conv_w"])),
                                         cat='Continuous')
        self.p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.config_data["p_max_bat_bike"]),
                                         cat='Continuous')
        self.p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.config_data["p_max_bat_bike"]),
                                         cat='Continuous')
        # Flags for bike battery
        self.flag_ch_bat_bike1 = pl.LpVariable.dicts('flag_ch_bat_bike1', (self.bikes,self.time),
                                                cat='Binary')
        self.flag_dc_bat_bike1 = pl.LpVariable.dicts('flag_dc_bat_bike1', (self.bikes,self.time),
                                                cat='Binary')
        self.flag_ch_bat_bike2 = pl.LpVariable.dicts('flag_ch_bat_bike2', (self.bikes,self.time),
                                                cat='Binary')
        self.flag_dc_bat_bike2 = pl.LpVariable.dicts('flag_dc_bat_bike2', (self.bikes,self.time),
                                                cat='Binary')
        # State Of Charge
        self.soc_bike = pl.LpVariable.dicts('soc_bike', (self.bikes, self.time),
                                       lowBound= float(self.config_data["soc_min_bike"]),
                                       upBound= float(self.config_data["soc_max_bike"]),
                                       cat='Continuous')
        # soc_min_otm_bike = pl.LpVariable('soc_min_otm_bike', lowBound=soc_min_bike, 
        #                                  upBound=soc_max_bike,
        #                                  cat='Continuous')

        # BATERIA ESTACIONÁRIA:
        self.p_ch_bat_est = pl.LpVariable.dicts('p_ch_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.config_data["p_max_bat_est"]),
                                           cat='Continuous')
        self.p_dc_bat_est = pl.LpVariable.dicts('p_dc_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.config_data["p_max_bat_est"]),
                                           cat='Continuous')
        self.flag_ch_bat_est = pl.LpVariable.dicts('flag_ch_bat_est', range(0,self.number_of_samples),
                                              cat='Binary')
        self.flag_dc_bat_est = pl.LpVariable.dicts('flag_dc_bat_est', range(0,self.number_of_samples),
                                              cat='Binary')
        self.dif_soc_ref_est = pl.LpVariable.dicts('dif_soc_ref_est', range(0,self.number_of_samples),
                                              lowBound=-1, upBound=1,
                                              cat='Continuous')
        self.mod_dif_soc_ref_est = pl.LpVariable.dicts('mod_dif_soc_ref_est', range(0,self.number_of_samples),
                                                  lowBound=0, upBound=1,
                                                  cat='Continuous')
        self.soc_est = pl.LpVariable.dicts('soc_est',
                                      range(0,self.number_of_samples),
                                      lowBound = float(self.config_data["soc_min_est"]),
                                      upBound = float(self.config_data["soc_max_est"]),
                                      cat='Continuous')

    def define_objective_funcion(self):
        self.prob += pl.lpSum([((self.p_rede[k]*self.sample_time*
                                 self.inicial_data.loc[k,'custo_energia'])/
                                (float(self.config_data["p_max_rede"]))*
                                self.sample_time*
                                (self.config_data["custo_max_energia"])*
                                self.config_data["peso_p_rede"] + 
                                self.mod_dif_soc_ref_est[k]*
                                self.config_data["peso_soc_est"] - 
                                ( pl.lpSum([self.soc_bike[bike][k]] 
                                           for bike in range(0,self.num_bikes)) )/
                                ( pl.lpSum([self.inicial_data.loc[k,'cx_bike_{}'.format(b)]]
                                           for b in range(0, self.num_bikes)) )*
                                self.config_data["peso_soc_bike"])
                               for k in self.inicial_data.index])
    
    def define_constraint(self):
        for k in self.inicial_data.index:
            
            # BATERIA BIKE:
            for bike in range(0, self.num_bikes):
                self.prob += self.p_ch_bike1[bike][k] == float(self.config_data["p_max_bat_bike"]) * self.flag_ch_bat_bike1[bike][k]
                self.prob += self.p_dc_bike1[bike][k] <= float(self.config_data["p_max_bat_bike"]) * self.flag_dc_bat_bike1[bike][k]
                self.prob += self.flag_ch_bat_bike1[bike][k] + self.flag_dc_bat_bike1[bike][k] <= 1 # simultaneity
                
                self.prob += self.p_ch_bike2[bike][k] <= float(self.config_data["p_max_bat_bike"])/float(self.config_data["eff_conv_w"]) * self.flag_ch_bat_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] <= float(self.config_data["p_max_bat_bike"]) * self.flag_dc_bat_bike2[bike][k]
                self.prob += self.flag_ch_bat_bike2[bike][k] + self.flag_dc_bat_bike2[bike][k] <= 1 # simultaneity
                
                # SOC
                # Inicio, então leitura
                if k == 0:
                    self.prob += self.soc_bike[bike][k] == self.inicial_data.loc[k,'soc_bike_{bike}'.format(bike=bike)]
                # Desconectada, então soc = 0
                elif self.inicial_data.loc[k,'cx_bike_{bike}'.format(bike=bike)] == 0:
                    self.prob += self.soc_bike[bike][k] == 0
                    self.prob += self.flag_ch_bat_bike1[bike][k] == 0
                    self.prob += self.flag_dc_bat_bike1[bike][k] == 0
                # Chegou agora, então leitura
                elif self.inicial_data.loc[k,'cx_bike_{bike}'.format(bike=bike)] == 1 and self.inicial_data.loc[k-1,'cx_bike_{bike}'.format(bike=bike)] == 0:
                    self.prob += self.soc_bike[bike][k] == self.inicial_data.loc[k,'soc_bike_{bike}'.format(bike=bike)] # Leitura do protocolo Modbus
                # Já estava conectada, então o algoritmo tem liberdade de controle
                else:
                    self.prob += self.soc_bike[bike][k] ==  self.soc_bike[bike][k-1] + (self.p_ch_bike1[bike][k-1] -
                                                             self.p_dc_bike1[bike][k-1])*self.sample_time/self.e_total_bat_bike"])
        
                # EFICIÊNICA Conversor Wireless
                self.prob += self.p_ch_bike1[bike][k] == float(self.config_data["eff_conv_w"]) * self.p_ch_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] == float(self.config_data["eff_conv_w"]) * self.p_dc_bike1[bike][k]
        
            # Pega o soc_min_otm_bike (Sugestao do Henry)
            # if k > 0:
            #     if (self.soc_bike[k] <= soc_min_otm_bike[k-1]):
            #         self.prob += soc_min_otm_bike[k] == self.soc_bike[k]
            
            # self.prob += soc_min_otm_bike <= self.soc_bike[k]
        
            # BATERIA ESTACIONÁRIA:
            self.prob += self.p_ch_bat_est[k] <= p_max_bat_est * flag_ch_bat_est[k]
            self.prob += self.p_dc_bat_est[k] <= p_max_bat_est * flag_dc_bat_est[k]             
            self.prob += flag_dc_bat_est[k] + flag_ch_bat_est[k] <= 1 # simultaneity
            # Calcula o módulo da distância do soc_est de sua referência
            self.prob += dif_soc_ref_est[k] == soc_ref_est - soc_est[k]
            if (dif_soc_ref_est[k] >= 0):
                self.prob += mod_dif_soc_ref_est[k] == dif_soc_ref_est[k]
            else:
                self.prob += mod_dif_soc_ref_est[k] == -dif_soc_ref_est[k]
        
            # SOC
            if k == 0:
                self.prob += soc_est[k] == soc_ini_est
            else:
                self.prob += soc_est[k] ==  soc_est[k-1] + (p_ch_bat_est[k-1] 
                                                        - p_dc_bat_est[k-1])*ts/e_total_bat_est
        
            # REDE
            self.prob += p_rede[k] == (p_rede_imp_ac[k] - p_rede_exp_ac[k])
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
            self.prob += p_rede_imp_ac[k] <= p_max_inv_ac * flag_rede_imp_ac[k]
            self.prob += p_rede_exp_ac[k] <= p_max_inv_ac * flag_rede_exp_ac[k]
            self.prob += flag_rede_imp_ac[k] + flag_rede_exp_ac[k] <= 1 # simultaneity
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
            self.prob += p_rede_imp_dc[k] <= p_max_inv_ac * flag_rede_imp_dc[k]
            self.prob += p_rede_exp_dc[k] <= p_max_inv_ac * flag_rede_exp_dc[k]
            self.prob += flag_rede_imp_dc[k] + flag_rede_exp_dc[k] <= 1 # simultaneity
        
            # EFICIÊNICA Conversor AC/DC
            self.prob += p_rede_exp_ac[k] == eff_conv_ac * p_rede_exp_dc[k]
            self.prob += p_rede_imp_dc[k] == eff_conv_ac * p_rede_imp_ac[k]
        
        
            # BALANÇO DE POTÊNCIA NO BARRAMENTO DC (Não considera a bat estacionária)
            self.prob += pl.lpSum([p_dc_bike2[bike][k]] for bike in range(0,num_bikes)) + self.inicial_data.loc[k,'potencia_PV'] + p_rede_imp_dc[
                k] + p_dc_bat_est[k] == pl.lpSum([p_ch_bike2[bike][k]] for bike in range(0,num_bikes)) + p_rede_exp_dc[k] + p_ch_bat_est[k]