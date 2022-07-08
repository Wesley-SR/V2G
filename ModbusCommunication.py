#!/usr/bin/env python3

import json
from pyModbusTCP.client import ModbusClient

class ModbusCommunication:
    def __init__(self, config_file_path):
        # Load initial datas
        with open(config_file_path, 'r') as json_file:
            self.config_data = json.load(json_file)

        # Init Modbus client
        try:
            self._modbus_client = ModbusClient(host = self.config_data['host'],
                            port=int(self.config_data['port']),
                            unit_id = int(self.config_data['client_id']),
                            debug=False, auto_open=True)
        except ValueError:
            print("Error with host or port params")
    
    def reed_modbus_data(self):
        return self._modbus_client.read_holding_registers(int(self.config_data['init_reg_read']), int(self.config_data['qtt_reg_read']))
    
    def write_modbus_data(self, data):
        self._modbus_client.write_multiple_registers(int(self.config_data['init_reg_write']),data)
        