#!/usr/bin/env python3


import time
import json
from ModbusCommunication import ModbusCommunication
from ModelPredictiveControl import ModelPredictiveControl
import pandas as pd

if __name__ == '__main__':
    
    path_to_modbus_settings = "configs_modbus.json"
    path_to_MPC_settings = "configs_mpc.json"
    
    # Init Modbus
    modbus_communication = ModbusCommunication(path_to_modbus_settings)
    new_data = modbus_communication.reed_modbus_data()
    print(new_data)
    
    # Init MPC
    mpc = ModelPredictiveControl(path_to_MPC_settings)
    

    # loop