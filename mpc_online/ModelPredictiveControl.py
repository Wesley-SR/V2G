#!/usr/bin/env python3

import json
import pandas as pd
import pulp as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import matplotlib.pyplot as plt




class ModelPredictiveControl:

    def __init__(self, path_to_config_file, path_to_inicial_data, path_to_control_signals):

        # LOAD CONFIGS
        with open(path_to_config_file, 'r') as json_file:
            self.config_data = json.load(json_file)
        print("1 - Read config files")

        # CREATE MATRIX TO STORE DATAS
        self.previous_data = pd.read_csv(path_to_inicial_data, sep = ";", index_col=['time'])
        self.forecast_data = self.previous_data
        self.control_signals = pd.read_csv(path_to_control_signals, sep = ";", index_col=['time'])

        # CREATE AN OBJECT TO THE OPTMIZATIOM PROBLEM
        self.prob = pl.LpProblem(self.config_data["optimization_name"], 
                                 pl.LpMinimize)

        # CHOSE THE SOLVER FOR THE OPTIMIZATION
        self.solver = pl.PULP_CBC_CMD()

        # CALL FUNCTIONS
        self.define_problem_parameters()
        self.define_problem_variables()
        print("2 - Define problem parameters and variables")




    def define_problem_parameters(self):
        
        # GENERIC PARAMETERS
        self.sample_time = float(self.config_data["sample_time"])
        self.number_of_samples = self.previous_data.index.size
        self.num_bikes = int(self.config_data["num_bikes"])
        self.bikes = range(0, self.num_bikes)
        self.time = range(0, self.number_of_samples+1)
        # EXTERNAL NETWORK
        self.net_pow_imp_max = float(self.config_data["net_pow_imp_max"])
        self.net_pow_exp_max = float(self.config_data["net_pow_exp_max"])
        self.energy_cost_imp_max = float(self.config_data["energy_cost_imp_max"])
        self.energy_cost_exp_max = float(self.config_data["energy_cost_exp_max"])
        # BIKE BATTERY
        self.bike_soc_max = float(self.config_data["bike_soc_max"]) # Maximum battery soc
        self.bike_soc_min = float(self.config_data["bike_soc_min"]) # Minimum battery soc
        self.bike_soc_ini = float(self.config_data["bike_soc_ini"])
        self.bike_pow_max = float(self.config_data["bike_pow_max"]) # (kW) -> ch/dc a 250 W
        self.bike_energy_max = float(self.config_data["bike_energy_max"]) # 6.3 # The total energy of the bikes battery (kWh)
        # OBS. 0.63 kWh per battery, being 12 V, são 52.5 Ah por bateria
        # STATIONARY BATTERY
        self.est_soc_max = float(self.config_data["est_soc_max"]) # soc Maximo da bateria
        self.est_soc_min = float(self.config_data["est_soc_min"]) # soc minimo da bateria
        self.est_soc_ini = float(self.config_data["est_soc_ini"])
        self.est_soc_ref = float(self.config_data["est_soc_ref"])
        self.est_pow_max = float(self.config_data["est_pow_max"]) # Deve ser o ch e dc constante (kW)
        self.est_energy_max = float(self.config_data["est_energy_max"]) # Energia total da bateria estacionária (kWh)

        # INVERTERS
        self.conv_w_eff = float(self.config_data["conv_w_eff"]) # 0.96 Eficiência conversor wireless
        self.conv_ac_eff =  float(self.config_data["conv_ac_eff"]) # 0.96
        self.inv_ac_pow_max = float(self.config_data["inv_ac_pow_max"]) # (kW)

        # WEIGHTS
        self.weight_soc_bike = float(self.config_data["weight_soc_bike"])
        self.weight_soc_est = float(self.config_data["weight_soc_est"])
        self.weight_pow_net = 1 - self.weight_soc_bike - self.weight_soc_est




    def define_problem_variables(self):

        
        self.net_pow_ac_imp = pl.LpVariable.dicts('net_pow_ac_imp', 
                                                 range(0, self.number_of_samples),
                                                 lowBound=0, 
                                                 upBound= float(self.net_pow_imp_max),
                                                 cat='Continuous')
        
        self.net_pow_ac_exp = pl.LpVariable.dicts('net_pow_ac_exp', 
                                                 range(0, self.number_of_samples),
                                                 lowBound=0,
                                                 upBound= float(self.net_pow_imp_max),
                                                 cat='Continuous')
        
        self.p_rede = pl.LpVariable.dicts('p_rede', range(0, self.number_of_samples),
                                     lowBound= - float(self.net_pow_imp_max),
                                     upBound= float(self.net_pow_imp_max),
                                     cat='Continuous')
        
        # Parte DC da rede
        self.net_pow_dc_imp = pl.LpVariable.dicts('net_pow_dc_imp', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.net_pow_imp_max),
                                            cat='Continuous')
        
        self.net_pow_dc_exp = pl.LpVariable.dicts('net_pow_dc_exp', range(0, int(self.number_of_samples)),
                                            lowBound = 0, upBound = float(self.net_pow_imp_max),
                                            cat='Continuous')

        # Flags for network import
        self.flag_net_ac_imp = pl.LpVariable.dicts('flag_net_ac_imp', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        self.flag_net_dc_imp = pl.LpVariable.dicts('flag_net_dc_imp', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        # Flags for network export
        self.flag_net_ac_exp = pl.LpVariable.dicts('flag_net_ac_exp', range(0, int(self.number_of_samples)),
                                               cat='Binary')
        self.flag_net_dc_exp = pl.LpVariable.dicts('flag_net_dc_exp', range(0, int(self.number_of_samples)),
                                               cat='Binary')

        # BIKE BATTERY:
        # Observation: (self.bikes, self.time) creates a variable with two dimensions 
        self.bike_pow_ch_1 = pl.LpVariable.dicts('bike_pow_ch_1', (self.bikes, self.time),
                                         lowBound = 0,
                                         upBound = float(self.bike_pow_max),
                                         cat='Continuous')
        self.p_ch_bike2 = pl.LpVariable.dicts('p_ch_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = (
                                             float(self.bike_pow_max)/
                                             float(self.conv_w_eff)),
                                         cat='Continuous')
        self.p_dc_bike1 = pl.LpVariable.dicts('p_dc_bike1', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.bike_pow_max),
                                         cat='Continuous')
        self.p_dc_bike2 = pl.LpVariable.dicts('p_dc_bike2', (self.bikes, self.time),
                                         lowBound = 0, upBound = float(self.bike_pow_max),
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
                                       lowBound= float(self.bike_soc_min),
                                       upBound= float(self.bike_soc_max),
                                       cat='Continuous')
        # soc_min_otm_bike = pl.LpVariable('soc_min_otm_bike', lowBound=bike_soc_min, 
        #                                  upBound=bike_soc_max,
        #                                  cat='Continuous')

        # BATERIA ESTACIONÁRIA:
        self.p_ch_bat_est = pl.LpVariable.dicts('p_ch_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.est_pow_max),
                                           cat='Continuous')
        self.p_dc_bat_est = pl.LpVariable.dicts('p_dc_bat_est', range(0,self.number_of_samples),
                                           lowBound=0, upBound = float(self.est_pow_max),
                                           cat='Continuous')
        self.flag_ch_bat_est = pl.LpVariable.dicts('flag_ch_bat_est', range(0,self.number_of_samples),
                                              cat='Binary')
        self.flag_dc_bat_est = pl.LpVariable.dicts('flag_dc_bat_est', range(0,self.number_of_samples),
                                              cat='Binary')
        self.dif_est_soc_ref = pl.LpVariable.dicts('dif_est_soc_ref', range(0,self.number_of_samples),
                                              lowBound=-1, upBound=1,
                                              cat='Continuous')
        self.mod_dif_est_soc_ref = pl.LpVariable.dicts('mod_dif_est_soc_ref', range(0,self.number_of_samples),
                                                  lowBound=0, upBound=1,
                                                  cat='Continuous')
        self.soc_est = pl.LpVariable.dicts('soc_est',
                                      range(0,self.number_of_samples),
                                      lowBound = float(self.est_soc_min),
                                      upBound = float(self.est_soc_max),
                                      cat='Continuous')




    def define_objective_funcion(self):
        self.prob += pl.lpSum([((self.p_rede[k]*self.sample_time*
                                 self.forecast_data.loc[k,'energy_cost_imp'])/
                                (float(self.net_pow_imp_max))*
                                self.sample_time*
                                (self.energy_cost_imp_max)*
                                self.weight_pow_net + 
                                self.mod_dif_est_soc_ref[k]*
                                self.weight_soc_est - 
                                (pl.lpSum([self.soc_bike[bike][k]] 
                                           for bike in range(0,self.num_bikes)) )/
                                (pl.lpSum([self.forecast_data.loc[k,'cx_bike_{}'.format(b)]]
                                           for b in range(0, self.num_bikes)) )*
                                self.weight_soc_bike)
                               for k in self.forecast_data.index])
        print("Define OF")




    def define_problem_constraint(self):
        for k in self.forecast_data.index:
            
            # BATERIA BIKE:
            for bike in range(0, self.num_bikes):
                self.prob += self.bike_pow_ch_1[bike][k] == float(self.bike_pow_max) * self.flag_ch_bat_bike1[bike][k]
                self.prob += self.p_dc_bike1[bike][k] <= float(self.bike_pow_max) * self.flag_dc_bat_bike1[bike][k]
                self.prob += self.flag_ch_bat_bike1[bike][k] + self.flag_dc_bat_bike1[bike][k] <= 1 # simultaneity
                
                self.prob += self.p_ch_bike2[bike][k] <= float(self.bike_pow_max)/float(self.conv_w_eff) * self.flag_ch_bat_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] <= float(self.bike_pow_max) * self.flag_dc_bat_bike2[bike][k]
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
                    self.prob += self.soc_bike[bike][k] ==  self.soc_bike[bike][k-1] + (self.bike_pow_ch_1[bike][k-1] - self.p_dc_bike1[bike][k-1])*self.sample_time/self.bike_energy_max
        
                # EFICIÊNICA Conversor Wireless
                self.prob += self.bike_pow_ch_1[bike][k] == float(self.conv_w_eff) * self.p_ch_bike2[bike][k]
                self.prob += self.p_dc_bike2[bike][k] == float(self.conv_w_eff) * self.p_dc_bike1[bike][k]
                
            # Pega o soc_min_otm_bike (Sugestao do Henry)
            # if k > 0:
            #     if (self.soc_bike[k] <= soc_min_otm_bike[k-1]):
            #         self.prob += soc_min_otm_bike[k] == self.soc_bike[k]
            
            # self.prob += soc_min_otm_bike <= self.soc_bike[k]
        
            # STATIONARY BATTERY
            self.prob += self.p_ch_bat_est[k] <= self.est_pow_max * self.flag_ch_bat_est[k]
            self.prob += self.p_dc_bat_est[k] <= self.est_pow_max * self.flag_dc_bat_est[k]             
            self.prob += self.flag_dc_bat_est[k] + self.flag_ch_bat_est[k] <= 1 # simultaneity
            # Calcula o módulo da distância do soc_est de sua referência
            self.prob += self.dif_est_soc_ref[k] == self.est_soc_ref - self.soc_est[k]
            if (self.dif_est_soc_ref[k] >= 0):
                self.prob += self.mod_dif_est_soc_ref[k] == self.dif_est_soc_ref[k]
            else:
                self.prob += self.mod_dif_est_soc_ref[k] == - self.dif_est_soc_ref[k]
                
            # SOC
            if k == 0:
                self.prob += self.soc_est[k] == self.est_soc_ini
            else:
                self.prob += self.soc_est[k] ==  self.soc_est[k-1] + (self.p_ch_bat_est[k-1] 
                                                        - self.p_dc_bat_est[k-1])*self.sample_time/self.est_energy_max
            
            # REDE
            self.prob += self.p_rede[k] == (self.net_pow_ac_imp[k] - self.net_pow_ac_exp[k])
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE AC
            self.prob += self.net_pow_ac_imp[k] <= self.inv_ac_pow_max * self.flag_net_ac_imp[k]
            self.prob += self.net_pow_ac_exp[k] <= self.inv_ac_pow_max * self.flag_net_ac_exp[k]
            self.prob += self.flag_net_ac_imp[k] + self.flag_net_ac_exp[k] <= 1 # simultaneity
            # IMPORTAÇÃO E EXPORTAÇÃO DA REDE - PARTE DC
            self.prob += self.net_pow_dc_imp[k] <= self.inv_ac_pow_max * self.flag_net_dc_imp[k]
            self.prob += self.net_pow_dc_exp[k] <= self.inv_ac_pow_max * self.flag_net_dc_exp[k]
            self.prob += self.flag_net_dc_imp[k] + self.flag_net_dc_exp[k] <= 1 # simultaneity
        
            # EFICIÊNICA Conversor AC/DC
            self.prob += self.net_pow_ac_exp[k] == self.conv_ac_eff * self.net_pow_dc_exp[k]
            self.prob += self.net_pow_dc_imp[k] == self.conv_ac_eff * self.net_pow_ac_imp[k]
        
            # BALANÇO DE POTÊNCIA NO BARRAMENTO DC (Não considera a bat estacionária)
            self.prob += pl.lpSum([self.p_dc_bike2[bike][k]] for bike in range(0, self.num_bikes)) 
            + self.forecast_data.loc[k,'pv_power'] 
            + self.net_pow_dc_imp[k] 
            + self.p_dc_bat_est[k] == pl.lpSum([self.p_ch_bike2[bike][k]] for bike in range(0,self.num_bikes)) 
            + self.net_pow_dc_exp[k] + self.p_ch_bat_est[k]
            
    
    
    
    def run_forecast_bike_behavior(self):
        # Use self.forecast_data
        pass
        
    
    

    def run_forecast_pv_generation(self):
        # CREAT AN ARRAY WITH PV GENERATION DATA
        local_previous_data = []
        for p in range(0, 144):
            local_previous_data.append([0])
            local_previous_data[p] = self.previous_data.loc[p, 'pv_power']
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
        
        self.forecast_data['pv_power'] = pd.DataFrame(forecast_pv)





    def set_new_data(self, measurement):
        # DISCARD THE FIRST AND DUPLICATE THE LAST SAMPLE
        self.previous_data[0:self.number_of_samples-2] = self.previous_data[0+1:self.number_of_samples-1] # [ ] Check later
        
        # UPDATE THE LAST SAMPLE
        self.previous_data[self.number_of_samples-1] = measurement[1]
       
        
        self.previous_data[self.number_of_samples-1,"pv_power"] = self.previous_data.loc[self.number_of_samples-1,"pv_power"]/1000
        self.previous_data[self.number_of_samples-1,"energy_cost_imp"] = self.previous_data.loc[self.number_of_samples-1,"energy_cost_imp"]/100
        self.previous_data[self.number_of_samples-1,"energy_cost_exp"] = self.previous_data.loc[self.number_of_samples-1,"energy_cost_exp"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_0"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_0"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_1"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_1"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_2"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_2"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_3"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_3"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_4"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_4"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_5"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_5"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_6"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_9"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_7"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_7"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_8"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_8"]/100
        self.previous_data[self.number_of_samples-1,"soc_bike_9"] = self.previous_data.loc[self.number_of_samples-1,"soc_bike_9"]/100
        
        # print(self.previous_data.loc[self.number_of_samples-1])

        




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




    def plot_results(self):
        time_array = self.previous_data.index
        
        p_rede_res = [0.0] * self.number_of_samples
        somatorio_p_rede = [0.0] * self.number_of_samples
        p_ch_bat_est_res = [0.0] * self.number_of_samples
        p_dc_bat_est_res = [0.0] * self.number_of_samples
        p_pv_res = [0.0] * self.number_of_samples
        soc_est_res = [0.0] * self.number_of_samples
        energy_cost_imp = [0.0] * self.number_of_samples
        mod_est = [0.0] * self.number_of_samples
        ch_dc_bike_res = np.zeros((self.num_bikes, self.number_of_samples))
        soc_bike_res = np.zeros((self.num_bikes, self.number_of_samples))
        
        for k in time_array:
            p_rede_res[k] = - self.p_rede[k].varValue
            p_pv_res[k] = self.previous_data.loc[k,'pv_power']
            energy_cost_imp[k] = self.previous_data.loc[k,'energy_cost_imp']
            soc_est_res[k] = self.soc_est[k].varValue
            mod_est[k] = self.mod_dif_est_soc_ref[k].varValue 
            
            for bike in range(0,self.num_bikes):
                ch_dc_bike_res[bike][k] = self.bike_pow_ch_1[bike][k].varValue - self.p_dc_bike1[bike][k].varValue
                soc_bike_res[bike][k] = self.soc_bike[bike][k].varValue
            if k >= 1:
                somatorio_p_rede[k] = (somatorio_p_rede[k-1] + p_rede_res[k])

        plt.figure()
        fig1,axs1 = plt.subplots(3)
        
        ''' p_rede e p_pv '''
        num_graf = 0
        # fig.suptitle('Penalidades: Bike {weight_soc_bike}, Est.: {weight_soc_est}, p_rede = {weight_pow_net} \n Efic. conv. wireless = {ef}'.
        #              format(weight_soc_bike=weight_soc_bike,weight_soc_est=weight_soc_est,weight_pow_net=weight_pow_net,ef=conv_w_eff),fontsize=10)
        axs1[num_graf].step(time_array/6, p_rede_res,c='#d62728',label='p_rede [kW] inversa')
        axs1[num_graf].step(time_array/6, p_pv_res,c='c',label='p_pv [kW]')
        axs1[num_graf].legend(loc='upper right',prop={'size': 7})
        axs1[num_graf].grid()
        axs1[num_graf].tick_params(axis='x', which='major', labelsize=5)
        axs1[num_graf].tick_params(axis='y', which='major', labelsize=10)
        axs1[num_graf].set_yticks([-10,-10,0,10,15])
        axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])
        
        ''' soc bikes '''
        num_graf +=1
        axs1[num_graf].step(time_array/6, soc_est_res,c='r',label='soc_est',linestyle = '--')
        for bike in range(0,self.num_bikes):
            axs1[num_graf].step(time_array/6, soc_bike_res[bike])#,label=('soc_bike_{bike}'.format(bike=bike)))
            # axs1[num_graf].legend(loc='lower right',prop={'size': 7})
        
        axs1[num_graf].legend(loc='lower right',prop={'size': 7})
        axs1[num_graf].grid()
        axs1[num_graf].tick_params(axis='x', which='major', labelsize=5)
        axs1[num_graf].tick_params(axis='y', which='major', labelsize=10)
        axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])
        
        ''' Soc da bateria estacionária '''
        num_graf +=1
        # axs1[num_graf].step(time_array/6, ch_dc_bike,c='#1f77b4',label='ch_dc_bike')
        # axs1[num_graf].step(time_array/6, soc_est_res,c='#1f77b4',label='soc_est')
        axs1[num_graf].step(time_array/6, energy_cost_imp,c='C1',label='energy_cost_imp [R$/(kWh)]')
        axs1[num_graf].legend(loc='lower center',prop={'size': 7})
        axs1[num_graf].set_ylabel('Amplitude')
        # axs1[1].set_xticks(time_array)
        axs1[num_graf].grid()
        axs1[num_graf].tick_params(axis='x', which='major', labelsize=7)
        axs1[num_graf].tick_params(axis='y', which='major', labelsize=10)
        axs1[num_graf].set_yticks([0,0.2,0.5,1])
        axs1[num_graf].set_xticks([0,5,6,8,10,12,16,17,18,20,21,22,25])
        
        ''' Somatório do p_rede [kWh] '''
        plot2 = plt.figure(2)
        fig2,axs2 = plt.subplots(1)
        plt.step(time_array/6, somatorio_p_rede,c='#d62728',label='somatorio_p_rede [kWh]')
        plt.legend(loc='lower right',prop={'size': 10})
        plt.grid()
        
        plt.show()



if __name__ == '__main__':
    mpc = ModelPredictiveControl("configs_mpc.json","dados_entrada.csv","control_signals.csv")
    