#!/usr/bin/env python3

import json
from pyModbusTCP.client import ModbusClient

class ModbusCommunication:
    def __init__(self, config_file_path):
        # Load initial datas
        with open(config_file_path, 'r') as json_file:
            self.config_data = json.load(json_file)
        
        # LOAD CONFIG PARAMETERS
        self._host = self.config_data['host']
        self._port = int(self.config_data['port'])
        self._client_id = int(self.config_data['client_id'])
        self._init_reg_read = int(self.config_data['init_reg_read'])
        self._qtt_reg_read = int(self.config_data['qtt_reg_read'])
        self._init_reg_write = int(self.config_data['init_reg_write'])
        
        # INIT MODBUS CLIENT
        try:
            self._modbus_client = ModbusClient(host = self._host, port= self._port,
                            unit_id = self._client_id, debug=False, auto_open=True)
            
            if self._modbus_client.read_holding_registers(0,1):
                print("\nModbus Client stated successfull\n")
            else:
                raise Exception("No found server")
            
        except Exception as error:
            print("\nError creating Modbus client: {}\n".format(error))
            raise
    
    def read_modbus_data(self):
        return self._modbus_client.read_holding_registers(self._init_reg_read, self._qtt_reg_read)
    
    def write_modbus_data(self, data):
        self._modbus_client.write_multiple_registers(self._init_reg_write, data)


if __name__ == '__main__':

    try:
        # Crete Modbus Communication
        modbus = ModbusCommunication('configs_modbus.json')
    
    except ValueError:
        print("Error: {}".format(ValueError))
    