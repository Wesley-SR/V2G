#!/usr/bin/env python3

# import ModbusCommunication
from ModbusCommunication import ModbusCommunication
from ModelPredictiveControl import ModelPredictiveControl
# from time import sleep

if __name__ == '__main__':
    
    continue_mpc = True
    counter = 0
    num_iterations = 1
    
    try:
        # Crete Modbus Communication
        modbus = ModbusCommunication('configs_modbus.json')
        
        # Create MPC object
        mpc = ModelPredictiveControl("configs_mpc.json",
                                     "dados_entrada_retirando_bikes.csv",
                                     "control_signals.csv")
    
        while(continue_mpc):
            
            # READ MEASUREMENT DATAS
            modbus_measurement = modbus.read_modbus_data()
    
            # RUN THE FORECAST ALGORITHMS
            mpc.set_new_data(modbus_measurement)
            
            # RUN THE OPTIMIZATION ALGORITHM
            mpc.run_mpc()
            
            # GET CONTROL SIGNALS
            control_signals = mpc.get_control_results()
            
            # WRITE THE CONTROL SIGNALS
            modbus.write_modbus_data(control_signals)
            
            #PLOT RESULTS
            mpc.plot_results()
            
            counter += 1
            if counter == num_iterations:
                continue_mpc = False
    
    except Exception as error:
        print("Error: {}".format(error))
    


        