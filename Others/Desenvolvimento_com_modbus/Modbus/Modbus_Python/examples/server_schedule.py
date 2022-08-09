#!/usr/bin/env python3

# Modbus/TCP server with start/stop schedule

# run this as root to listen on TCP priviliged ports (<= 1024)
# default Modbus/TCP port is 502 so we prefix call with sudo (avoid Errno 13)
#
#   sudo ./server_shedule.py --host 0.0.0.0

import argparse
import time
from pyModbusTCP.server import ModbusServer
# need https://github.com/dbader/schedule
import schedule


# word @0 = second since 00:00 divide by 10 to avoid 16 bits overflow
def alive_word_job():
    server.data_bank.set_holding_registers(0, [int(time.time()) % (24*3600) // 10])


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, default='localhost', help='Host (default: localhost)')
    parser.add_argument('-p', '--port', type=int, default=502, help='TCP port (default: 502)')
    args = parser.parse_args()
    # init modbus server and start it
    server = ModbusServer(host=args.host, port=args.port, no_block=True)
    server.start()
    # init scheduler
    # schedule a daily downtime (from 18:00 to 06:00)
    schedule.every().day.at('18:00').do(server.stop)
    schedule.every().day.at('06:00').do(server.start)
    # update life word at @0
    schedule.every(10).seconds.do(alive_word_job)
    # main loop
    while True:
        schedule.run_pending()
        time.sleep(1)
