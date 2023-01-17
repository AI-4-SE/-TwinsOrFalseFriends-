#!/usr/bin/env python3

import time
import socket
from famefetcher import SocketSender
import pandas as df

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 12347  # Port to listen on (non-privileged ports are > 1023)


def recvall(sock, length=None):
    BUFF_SIZE = 2 ** 10  # 4096   # 4 KiB
    data = b''
    if length is None:
        while True:
            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
            # time.sleep(0.1)
    else:
        remaining_bytes = length
        for i in range(0, length, BUFF_SIZE):
            part = sock.recv(BUFF_SIZE)
            data += part
    return data


def rec_msg_length(sock, BUFF_SIZE=64):
    data = sock.recv(BUFF_SIZE)
    return data


def main():
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    with socket.socket() as s:
        s.bind((HOST, PORT))
        s.listen()

        print("Waiting for connection.")
        conn, addr = s.accept()
        print("Connected.")
        with conn:
            while True:
                print('Connected by', addr)
                print("Waiting for Msg Length")
                data = w8_4_msg_len(conn)
                msg_len = int.from_bytes(data, byteorder='big', signed=False)
                print("Waiting for Data of {} bytes".format(msg_len))
                data = w8_4_data(conn, length=msg_len)
                # while data:
                #     print(data)
                #     # data_arr.append(data)
                #     data = conn.recv(1024)

                print("Received Data (", len(data), "bytes )")
                # if not data:
                #     break
                df = SocketSender.decompress(data)
                print(len(df), list(df))
                print("Sending ok.")
                conn.sendall('OK.'.encode('ascii'))


def w8_4_data(conn, length=None):
    data = recvall(conn, length)
    while len(data) == 0:
        time.sleep(0.1)
        data = recvall(conn, length)
    return data


def w8_4_msg_len(conn):
    data = rec_msg_length(conn)
    while len(data) == 0:
        time.sleep(0.1)
        data = rec_msg_length(conn)
    return data


if __name__ == "__main__":
    main()
