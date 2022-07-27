#!/usr/bin/env python3

# import ModbusCommunication
from ModbusCommunication import ModbusCommunication
from ModelPredictiveControl import ModelPredictiveControl
from time import sleep

if __name__ == '__main__':
    
    continue_mpc = True
    
    try:
        # Crete Modbus Communication
        modbus = ModbusCommunication('configs_modbus.json')
        
        # Create MPC object
        mpc = ModelPredictiveControl("configs_mpc.json","dados_entrada_retirando_bikes.csv","control_signals.csv")
    
    except ValueError:
        print("Error: {}".format(ValueError))
        
    # measurement = Dados históricos medidos (Matriz de 144 linhas)
    # forecast_use_bikes = matriz 144 posicoes
    # forecast_pv_generation = matriz 14 posicoes
    # optimization_results = resultado do algoritmo de otimização (Matriz de 144 linhas), vem a cada 10 minutos
    # control_signal = Sinal de controle (vetor de 1 linha), é aplicado a cada 10 minutos
    
    # while(continue_mpc):
        
        # READ MEASUREMENT DATAS
        modbus_measurement = modbus.read_modbus_data() # modbus_measurement = [PV P_R soc_est cx_bike1 ... cx_bike10 soc_bike1 ... soc_bike10] = Matrix 1x24
        
        # RUN THE FORECAST ALGORITHMS
        mpc.set_new_data(modbus_measurement)
        
        # RUN THE OPTIMIZATION ALGORITHM
        mpc.run_mpc()
        
        # GET CONTROL SIGNALS
        control_signals = mpc.get_control_results()
        
        # WRITE THE CONTROL SIGNALS
        modbus.write_modbus_data(control_signals)
        
        # sleep(10*60)