#!/usr/bin/env python3

# import ModbusCommunication

from ModelPredictiveControl import ModelPredictiveControl
# from time import sleep

if __name__ == '__main__':
    
    continue_mpc = True
    counter = 0
    num_iterations = 1
    
    try:
        
        # CREATE MPC OBJECT
        mpc = ModelPredictiveControl("configs_mpc.json",
                                     "dados_entrada.csv",
                                     "control_signals.csv")
        
        # RUN OFFLINE OPTMIZATION
        # mpc.run_offline_optimization()
        
        while(continue_mpc):
            
            # READ MEASUREMENT DATAS
            
            
            # RUN THE FORECAST ALGORITHMS
            
            
            # RUN THE OPTIMIZATION ALGORITHM
            mpc.run_mpc()
            
            # GET CONTROL SIGNALS
            control_signals = mpc.get_control_results()
            
            # WRITE THE CONTROL SIGNALS
            
            
            #PLOT RESULTS
            mpc.plot_results()
             
            counter += 1
            if counter == num_iterations:
                continue_mpc = False
    
    except Exception as error:
        print("Error: {}".format(error))
    


        