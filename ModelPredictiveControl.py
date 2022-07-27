#!/usr/bin/env python3

import json
import pandas as pd
import pulp as pl
# import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
# import matplotlib.pyplot as plt




class ModelPredictiveControl:

    def __init__(self, path_to_config_file, path_to_inicial_data, path_to_control_signals):

        # LOAD CONFIGS
        with open(path_to_config_file, 'r') as json_file:
            self.config_data = json.load(json_file)

        # CREATE MATRIX TO STORE DATAS
        self.previous_data = pd.read_csv(path_to_inicial_data, sep = ";", index_col=['tempo'])
        self.forecast_data = self.previous_data
        self.control_signals = pd.read_csv(path_to_control_signals, sep = ";", index_col=['tempo'])

        # CREATE AN OBJECT TO THE OPTMIZATIOM PROBLEM
        self.prob = pl.LpProblem(self.config_data["optimization_name"], 
                                 pl.LpMinimize)

        # CHOSE THE SOLVER FOR THE OPTIMIZATION
        self.solver = pl.PULP_CBC_CMD()

        # CALL FUNCTIONS
        self.define_problem_parameters()
        self.define_problem_variables()




    def define_problem_parameters(self):
        
        # GENERIC PARAMETERS
        self.sample_time = float(self.config_data["sample_time"])
        self.number_of_samples = self.previous_data.index.size
        self.num_bikes = int(self.config_data["num_bikes"])
        self.bikes = range(0, self.num_bikes)
        self.time = range(0, self.number_of_samples+1)
        # EXTERNAL NETWORK
        self.p_max_rede = float(self.config_data["p_max_rede"])
        self.max_energy_cost = float(self.config_data["max_energy_cost"])
        # BIKE BATTERY
        self.soc_max_bike = float(self.config_data["soc_max_bike"]) # Maximum battery soc
        self.soc_min_bike = float(self.config_data["soc_min_bike"]) # Minimum battery soc
        self.soc_ini_bike = float(self.config_data["soc_ini_bike"])
        self.p_max_bat_bike = float(self.config_data["p_max_bat_bike"]) # (kW) -> ch/dc a 250 W
        self.e_total_bat_bike = float(self.config_data["e_total_bat_bike"]) # 6.3 # The total energy of the bikes battery (kWh)
        # OBS. 0.63 kWh per battery, being 12 V, são 52.5 Ah por bateria
        # STATIONARY BATTERY
        self.soc_max_est = float(self.config_data["soc_max_est"]) # soc Maximo da bateria
        self.soc_min_est = float(self.config_data["soc_min_est"]) # soc minimo da bateria
        self.soc_ini_est = float(self.config_data["soc_ini_est"])
        self.p_max_bat_est = float(self.config_data["p_max_bat_est"]) # Deve ser o ch e dc constante (kW)
        self.e_total_bat_est = float(self.config_data["e_total_bat_est"]) # Energia total da bateria estacionária (kWh)
        self.soc_ref_est = float(self.config_data["soc_ref_est"])
        # INVERTERS
        self.eff_conv_w = float(self.config_data["eff_conv_w"]) # 0.96 Eficiência conversor wireless
        self.eff_conv_ac =  float(self.config_data["eff_conv_ac"]) # 0.96
        self.p_max_inv_ac = float(self.config_data["p_max_inv_ac"]) # (kW)
        self.q_max_inv_ac = float(self.config_data["q_max_inv_ac"]) # (kvar)
        # WEIGHTS
        self.peso_soc_bike = float(self.config_data["peso_soc_bike"])
        self.peso_soc_est = float(self.config_data["peso_soc_est"])
        self.peso_p_rede = 1 - self.peso_soc_bike - self.peso_soc_est




    def define_problem_variables(self):

        self.p_rede_exp_ac = pl.LpVariable.dicts('p_rede_exp_ac', 
                                                 range(0, self.number_of_samples),
                                                 lowBound=0,
                                                 upBound= float(self.p_max_rede),
                                                 cat='Continuous')
        
        self.p_rede_imp_ac = pl.LpVariable.dicts('p_rede_imp_ac', 
                                                 range(0, self.number_of_samples),
                                                 lowBound=0, 
                                                 upBound= float(self.p_max_rede),
                                                 cat='Continuous')
        
        self.p_rede = pl.LpVariable.dicts('p_rede', range(0, self.number_of_samples),
                                     lowBound= - float(self.p_max_rede),
                                     upBound= float(self.p_max_rede),
                                     cat='Continuous')
        
        # q_rede_exp_ac = pl.LpVariable.dicts('q_rede_exp_ac', range(0,self.number_of_samples),
        #                                     lowBound=0, upBound=p_max_rede,
        #                                     cat='Continuous')
        # q_rede_imp_ac = pl.LpVariable.dicts('q_rede_imp_ac', range(0,self.number_of_samples),
        #                                     lowBound=0, upBound=p_max_rede,
        #                                     cat='Continuous')

        # Parte DC da rede
        self.p_rede_exp_dc = pl.LpVariable.dicts('p_rede_exp_dc', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.p_max_rede),
                                            cat='Continuous')
        self.p_rede_imp_dc = pl.LpVariable.dicts('p_rede_imp_dc', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.p_max_rede),
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

        # BIKE BATTERY:
        # Observation: (self.bikes, self.time) creates a variable with two dimensions 
        self.p_ch_bike1 = pl.LpVariable.dicts('p_ch_bike1', (self.bikes, self.time),
                                         lowBound = 0,
                                         upBound = float(self.p_max_bat_bike),
                                         cat='Continuous')
        self.p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = (
                                             float(self.p_max_bat_bike)/
                                             float(self.eff_conv_w)),
                                         cat='Continuous')
        self.p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.p_max_bat_bike),
                                         cat='Continuous')
        self.p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.p_max_bat_bike),
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
                                       lowBound= float(self.soc_min_bike),
                                       upBound= float(self.soc_max_bike),
                                       cat='Continuous')
        # soc_min_otm_bike = pl.LpVariable('soc_min_otm_bike', lowBound=soc_min_bike, 
        #                                  upBound=soc_max_bike,
        #                                  cat='Continuous')

        # BATERIA ESTACIONÁRIA:
        self.p_ch_bat_est = pl.LpVariable.dicts('p_ch_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.p_max_bat_est),
                                           cat='Continuous')
        self.p_dc_bat_est = pl.LpVariable.dicts('p_dc_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.p_max_bat_est),
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
                                      lowBound = float(self.soc_min_est),
                                      upBound = float(self.soc_max_est),
                                      cat='Continuous')




    def define_objective_funcion(self):
        self.prob += pl.lpSum([((self.p_rede[k]*self.sample_time*
                                 self.forecast_data.loc[k,'custo_energia'])/
                                (float(self.p_max_rede))*
                                self.sample_time*
                                (self.max_energy_cost)*
                                self.peso_p_rede + 
                                self.mod_dif_soc_ref_est[k]*
                                self.peso_soc_est - 
                                (pl.lpSum([self.soc_bike[bike][k]] 
                                           for bike in range(0,self.num_bikes)) )/
                                (pl.lpSum([self.forecast_data.loc[k,'cx_bike_{}'.format(b)]]
                                           for b in range(0, self.num_bikes)) )*
                                self.peso_soc_bike)
                               for k in self.forecast_data.index])




    def define_problem_constraint(self):
        for k in self.forecast_data.index:
            
            # BATERIA BIKE:
            for bike in range(0, self.num_bikes):
                self.prob += self.p_ch_bike1[bike][k] == float(self.p_max_bat_bike) * self.flag_ch_bat_bike1[bike][k]
                self.prob += self.p_dc_bike1[bike][k] <= float(self.p_max_bat_bike) * self.flag_dc_bat_bike1[bike][k]
                self.prob += self.flag_ch_bat_bike1[bike][k] + self.flag_dc_bat_bike1[bike][k] <= 1 # simultaneity
                
                self.prob += self.p_ch_bike2[bike][k] <= float(self.p_max_bat_bike)/float(self.eff_conv_w) * self.flag_ch_bat_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] <= float(self.p_max_bat_bike) * self.flag_dc_bat_bike2[bike][k]
                self.prob += self.flag_ch_bat_bike2[bike][k] + self.flag_dc_bat_bike2[bike][k] <= 1 # simultaneity
                
                # SOC
                # Inicio, então leitura
                if k == 0:
                    self.prob += self.soc_bike[bike][k] == self.forecast_data.loc[k,'soc_bike_{bike}'.format(bike=bike)]
                # Desconectada, então soc = 0
                elif self.forecast_data.loc[k,'cx_bike_{bike}'.format(bike=bike)] == 0:
                    self.prob += self.soc_bike[bike][k] == 0
                    self.prob += self.flag_ch_bat_bike1[bike][k] == 0
                    self.prob += self.flag_dc_bat_bike1[bike][k] == 0
                # Chegou agora, então leitura
                elif self.forecast_data.loc[k,'cx_bike_{bike}'.format(bike=bike)] == 1 and self.forecast_data.loc[k-1,'cx_bike_{bike}'.format(bike=bike)] == 0:
                    self.prob += self.soc_bike[bike][k] == self.forecast_data.loc[k,'soc_bike_{bike}'.format(bike=bike)] # Leitura do protocolo Modbus
                # Já estava conectada, então o algoritmo tem liberdade de controle
                else:
                    self.prob += self.soc_bike[bike][k] ==  self.soc_bike[bike][k-1] + (self.p_ch_bike1[bike][k-1] - self.p_dc_bike1[bike][k-1])*self.sample_time/self.e_total_bat_bike
        
                # EFICIÊNICA Conversor Wireless
                self.prob += self.p_ch_bike1[bike][k] == float(self.eff_conv_w) * self.p_ch_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] == float(self.eff_conv_w) * self.p_dc_bike1[bike][k]
                
            # Pega o soc_min_otm_bike (Sugestao do Henry)
            # if k > 0:
            #     if (self.soc_bike[k] <= soc_min_otm_bike[k-1]):
            #         self.prob += soc_min_otm_bike[k] == self.soc_bike[k]
            
            # self.prob += soc_min_otm_bike <= self.soc_bike[k]
        
            # STATIONARY BATTERY
            self.prob += self.p_ch_bat_est[k] <= self.p_max_bat_est * self.flag_ch_bat_est[k]
            self.prob += self.p_dc_bat_est[k] <= self.p_max_bat_est * self.flag_dc_bat_est[k]             
            self.prob += self.flag_dc_bat_est[k] + self.flag_ch_bat_est[k] <= 1 # simultaneity
            # Calcula o módulo da distância do soc_est de sua referência
            self.prob += self.dif_soc_ref_est[k] == self.soc_ref_est - self.soc_est[k]
            if (self.dif_soc_ref_est[k] >= 0):
                self.prob += self.mod_dif_soc_ref_est[k] == self.dif_soc_ref_est[k]
            else:
                self.prob += self.mod_dif_soc_ref_est[k] == - self.dif_soc_ref_est[k]
                
            # SOC
            if k == 0:
                self.prob += self.soc_est[k] == self.soc_ini_est
            else:
                self.prob += self.soc_est[k] ==  self.soc_est[k-1] + (self.p_ch_bat_est[k-1] 
                                                        - self.p_dc_bat_est[k-1])*self.sample_time/self.e_total_bat_est
            
            # REDE
            self.prob += self.p_rede[k] == (self.p_rede_imp_ac[k] - self.p_rede_exp_ac[k])
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
            self.prob += self.p_rede_imp_ac[k] <= self.p_max_inv_ac * self.flag_rede_imp_ac[k]
            self.prob += self.p_rede_exp_ac[k] <= self.p_max_inv_ac * self.flag_rede_exp_ac[k]
            self.prob += self.flag_rede_imp_ac[k] + self.flag_rede_exp_ac[k] <= 1 # simultaneity
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
            self.prob += self.p_rede_imp_dc[k] <= self.p_max_inv_ac * self.flag_rede_imp_dc[k]
            self.prob += self.p_rede_exp_dc[k] <= self.p_max_inv_ac * self.flag_rede_exp_dc[k]
            self.prob += self.flag_rede_imp_dc[k] + self.flag_rede_exp_dc[k] <= 1 # simultaneity
        
            # EFICIÊNICA Conversor AC/DC
            self.prob += self.p_rede_exp_ac[k] == self.eff_conv_ac * self.p_rede_exp_dc[k]
            self.prob += self.p_rede_imp_dc[k] == self.eff_conv_ac * self.p_rede_imp_ac[k]
        
            # BALANÇO DE POTÊNCIA NO BARRAMENTO DC (Não considera a bat estacionária)
            self.prob += pl.lpSum([self.p_dc_bike2[bike][k]] for bike in range(0, self.num_bikes)) 
            + self.forecast_data.loc[k,'potencia_PV'] 
            + self.p_rede_imp_dc[k] 
            + self.p_dc_bat_est[k] == pl.lpSum([self.p_ch_bike2[bike][k]] for bike in range(0,self.num_bikes)) 
            + self.p_rede_exp_dc[k] + self.p_ch_bat_est[k]
            
    
    
    
    def run_forecast_bike_behavior(self):
        # Use self.forecast_data
        pass
        
    
    

    def run_forecast_pv_generation(self):
        # CREAT AN ARRAY WITH PV GENERATION DATA
        local_previous_data = []
        for p in range(0, 144):
            local_previous_data.append([0])
            local_previous_data[p] = self.previous_data.loc[p, 'potencia_PV']
        # DO THE TRANSPOSE OF THE ARRAY
        local_previous_data = np.array([local_previous_data],dtype=str).T
        # LOAD THE NEURAL NETWORK STRUCTURE
        forecast_model = open('previsao_PV_10_min.json', 'r')
        estrutura_rede_PV = forecast_model.read()
        forecast_model.close() 
        classificador_PV = model_from_json(estrutura_rede_PV)
        # LOAD THE PARAMETERS OF NEURAL NETWORK
        classificador_PV.load_weights('LSTM_PV_144_2.h5') 
        
        n_steps_in = 144
        n_features = 1
        
        # IT NORMALIZE INPUT DATA
        scaler = MinMaxScaler(feature_range=(0, 1))
        previous_pv_data = scaler.fit_transform(local_previous_data)
        
        # DEIXAR NA FORMA: (1,144,1)
        x_input_1 = previous_pv_data.reshape((1, n_steps_in, n_features)) 
        
        # FAZER A PREVISÃO:
        forecast_pv = classificador_PV.predict(x_input_1, verbose=0) 
        
        # DEIXAR A PREVISÃO EM kW:
        forecast_pv = scaler.inverse_transform(forecast_pv).T
        
        self.forecast_data['potencia_PV'] = pd.DataFrame(forecast_pv)





    def set_new_data(self, measurement):
        # DISCARD THE FIRST AND DUPLICATE THE LAST SAMPLE
        self.previous_data[0:self.number_of_samples-2] = self.previous_data[0+1:self.number_of_samples-1] # [ ] Check later
        
        # UPDATE THE LAST SAMPLE
        self.previous_data[self.number_of_samples] = measurement[1]




    def get_control_results(self):
        
        control_signals = self.control_signals[:1]
        control_signals = self.control_signals.iloc[1].values # [] Check with float number, because modbus is only int
        return control_signals




    def run_mpc(self):
        self.run_forecast_bike_behavior()
        self.run_forecast_pv_generation()
        self.define_problem_constraint()
        self.define_objective_funcion()
        self.prob.solve(self.solver)




if __name__ == '__main__':
    mpc = ModelPredictiveControl("configs_mpc.json","dados_entrada_retirando_bikes.csv","control_signals.csv")
    