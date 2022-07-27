#!/usr/bin/env python3

# Modbus/TCP server
#
# run this as root to listen on TCP priviliged ports (<= 1024)
# add "--host 0.0.0.0" to listen on all available IPv4 addresses of the host

# import argparse
from pyModbusTCP.server import ModbusServer
from time import sleep

if __name__ == '__main__':

    # parse args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-H', '--host', type=str, default='localhost', help='Host (default: localhost)')
    # parser.add_argument('-p', '--port', type=int, default=502, help='TCP port (default: 502)')
    # args = parser.parse_args()
    IP_server = '127.1.0.0'
    # IP_server = '192.168.0.7'
    porta =  502

    print("host= {}".format(IP_server))
    print("port= {}".format(porta))
    
    try:
        # Cria o servidor
        server = ModbusServer(host=IP_server, port=porta)
        # Inicializa o servidor
        print("Start server \n")
        server.start()
        
        # state = [0]
        # tempo = 1
        # DataBank.set_words(0, 0)
        # while True:
        #     if state != DataBank.get_words(1):
        #         state = DataBank.get_words(1)
        #         print("Value of Register 1 has changed to " +str(state))
        #     sleep(1)
        #     print("Tempo: {tempo}s".format(tempo=tempo))

    except Exception as error:
        print(error)
        print("Shutdown server \n")
        server.stop()
        print("Server is offline")
