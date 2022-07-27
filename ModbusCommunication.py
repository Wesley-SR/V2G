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
        except Exception as error:
            print("Error creating Modbus client \n Error: {}".format(error))
    
    def read_modbus_data(self):
        return self._modbus_client.read_holding_registers(self._init_reg_read, self._qtt_reg_read)
    
    def write_modbus_data(self, data):
        self._modbus_client.write_multiple_registers(self._init_reg_write, data)
        