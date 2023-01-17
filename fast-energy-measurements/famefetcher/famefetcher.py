import pandas as pd

# pyserial not serial
from serial import Serial
import csv
import time
import numpy as np
import sys
from datetime import datetime

from bitstring import BitArray
from serial.tools.list_ports import comports as comports
from copy import deepcopy

from bitarray import bitarray
from bitarray.util import int2ba

from yaml import load
from enum import Enum

from typing import List
import socket
import pickle

from mqttClients import MQTTClient
from mqttClients import MQTTPublisher
import argparse


class LogicLevel(Enum):
    GND = "GND"
    VS = "VS"
    SDA = "SDA"
    SCL = "SCL"


# Class for serial bus addresses in I²C logic. For reference see linked TI INA226 p.18, table 2.
class I2CAddress:
    # all possible pin address combinations and their corresponding slave adresses
    ADDRESSES = {
        (LogicLevel.GND, LogicLevel.GND): [1, 0, 0, 0, 0, 0, 0],
        (LogicLevel.GND, LogicLevel.VS): [1, 0, 0, 0, 0, 0, 1],
        (LogicLevel.GND, LogicLevel.SDA): [1, 0, 0, 0, 0, 1, 0],
        (LogicLevel.GND, LogicLevel.SCL): [1, 0, 0, 0, 0, 1, 1],

        (LogicLevel.VS, LogicLevel.GND): [1, 0, 0, 0, 1, 0, 0],
        (LogicLevel.VS, LogicLevel.VS): [1, 0, 0, 0, 1, 0, 1],
        (LogicLevel.VS, LogicLevel.SDA): [1, 0, 0, 0, 1, 1, 0],
        (LogicLevel.VS, LogicLevel.SCL): [1, 0, 0, 0, 1, 1, 1],

        (LogicLevel.SDA, LogicLevel.GND): [1, 0, 0, 1, 0, 0, 0],
        (LogicLevel.SDA, LogicLevel.VS): [1, 0, 0, 1, 0, 0, 1],
        (LogicLevel.SDA, LogicLevel.SDA): [1, 0, 0, 1, 0, 1, 0],
        (LogicLevel.SDA, LogicLevel.SCL): [1, 0, 0, 1, 0, 1, 1],

        (LogicLevel.SCL, LogicLevel.GND): [1, 0, 0, 1, 1, 0, 0],
        (LogicLevel.SCL, LogicLevel.VS): [1, 0, 0, 1, 1, 0, 1],
        (LogicLevel.SCL, LogicLevel.SDA): [1, 0, 0, 1, 1, 1, 0],
        (LogicLevel.SCL, LogicLevel.SCL): [1, 0, 0, 1, 1, 1, 1],
    }

    def __init__(self, a1, a0):
        if a1 not in LogicLevel:
            raise ValueError(
                "{} is not a valid logic levels. Please choose from: {}".format(
                    a1, ", ".join("I2CAddress.{}".format(level) for level in LogicLevel)))
        if a0 not in LogicLevel:
            raise ValueError(
                "{} is not a valid logic levels. Please choose from: {}".format(
                    a0, ", ".join("I2CAddress.{}".format(level) for level in LogicLevel)))

        self.a1 = a1
        self.a0 = a0

    def get_bits(self) -> bitarray:
        address = I2CAddress.ADDRESSES[self.a1, self.a0]
        ba = bitarray()
        ba.extend(address)
        return ba


# Class for INA configuration: mapping of ina IDs, I²C addresses, busses and physical properties
class INA:
    U = 0.0819
    CONVERSION_TIMES = {
        140: [0, 0, 0],
        204: [0, 0, 1],
        332: [0, 1, 0],
        588: [0, 1, 1],
        1100: [1, 0, 0],
        2116: [1, 0, 1],
        4156: [1, 1, 0],
        8244: [1, 1, 1],
    }

    AVERAGING_SIZES = {
        1: [0, 0, 0],
        4: [0, 0, 1],
        16: [0, 1, 0],
        64: [0, 1, 1],
        128: [1, 0, 0],
        256: [1, 0, 1],
        512: [1, 1, 0],
        1024: [1, 1, 1],
    }

    def write_config_calib(self, write_buff):
        id_byte = self.device_id.encode("ascii")
        bus_bits = bits_fill_left(int2ba(self.bus_num))
        bus_byte = bus_bits.tobytes()

        conf_bytes = bits_fill_left(self.get_configuration_bytes()).tobytes()
        calib_bytes = bits_fill_left(self.get_calibration_bytes()).tobytes()
        i2c_address = bits_fill_left(self.address.get_bits()).tobytes()

        write_buff.write(id_byte)
        write_buff.write(bus_byte)
        write_buff.write(i2c_address)
        write_buff.write(calib_bytes)
        write_buff.write(conf_bytes)

    def __init__(self, address: I2CAddress, device_id, bus_num,
                 conversion_time_current, conversion_time_voltage, averaging, r_shunt, ina_long_id):
        self.device_id = device_id
        self.ina_long_id = ina_long_id
        if bus_num > 2 or bus_num < 0:
            raise ValueError(
                "Only I2C busses 0, 1, and 2 exist - you chose {} - "
                "Please chose one of the three existing.".format(bus_num))
        self.bus_num = bus_num
        self.address = address

        if conversion_time_current not in INA.CONVERSION_TIMES:
            raise ValueError(
                "Could not find shunt voltage / current conversion time {} in valid conversion times: {}".format(
                    conversion_time_current, INA.CONVERSION_TIMES))

        if conversion_time_voltage not in INA.CONVERSION_TIMES:
            raise ValueError(
                "Could not find bus voltage / voltage conversion time {} in valid conversion times: {}".format(
                    conversion_time_voltage, INA.CONVERSION_TIMES))

        self.conversion_time_current = conversion_time_current
        self.conversion_time_current_bits = INA.CONVERSION_TIMES[conversion_time_current]
        self.conversion_time_voltage = conversion_time_voltage
        self.conversion_time_voltage_bits = INA.CONVERSION_TIMES[conversion_time_voltage]

        if averaging not in INA.AVERAGING_SIZES:
            raise ValueError(
                "Could not find average number {} in valid average numbers: {}".format(
                    averaging, INA.AVERAGING_SIZES))
        self.averaging = averaging
        self.averaging_bits = INA.AVERAGING_SIZES[averaging]
        self.r_shunt = r_shunt
        self.max_current = self.calc_max_current(self.r_shunt)
        self.current_LSB, self.power_LSB = self.get_current_lsb()

        self.calib_register = self.get_calibration_bytes()
        self.conf_register = self.get_configuration_bytes()

    def get_bytes_from_int(self, i: int):
        bts = i.to_bytes(2, byteorder='big', signed=False)
        return bts

    def compute_calibration(self):
        calibration = 0.00512 / (self.current_LSB * self.r_shunt)
        int_calib = int(calibration)
        return int_calib

    def get_current_lsb(self):
        current_LSB = self.max_current / 32767
        return current_LSB, current_LSB * 25

    def get_power(self, power_register: int):
        return power_register * self.power_LSB

    def get_calibration_bytes(self) -> bitarray:
        # left-most bit D15 is ignored
        calibration = self.compute_calibration()
        ba = int2ba(int(calibration))
        return ba

    def get_configuration_bytes(self) -> bitarray:
        bts = bitarray(endian="big")
        bts.extend([0, 1, 0, 0])  # default values from docs
        mode_bits = bitarray(3, endian="little")
        mode_bits.setall(1)

        bts.extend(self.averaging_bits)
        bts.extend(self.conversion_time_voltage_bits)
        bts.extend(self.conversion_time_current_bits)
        bts.extend(mode_bits)

        return bts

    def __repr__(self):
        s = "@Bus#{} <{}> {} ".format(self.bus_num, self.device_id, self.ina_long_id)
        return s

    def __str__(self):
        return self.__repr__()

    def calc_max_current(self, r_shunt):
        I = INA.U / r_shunt
        return I


class YamlConfig:
    def __init__(self, config_path):
        self.conf = load(open(config_path))
        self.yaml_instances = self.conf["instances"]
        defaults = self.conf["defaults"]
        broker = self.conf["broker"]
        self.instance_configs = {key: InstanceConfig(inst, broker, defaults) for key, inst in
                                 self.yaml_instances.items()}

    def get_instance(self, instance_name):
        return self.instance_configs[instance_name]


class InstanceConfig:
    def __init__(self, instance, broker, defaults):
        self.conf = instance
        self.i2c_speed: int = self.conf["i2c_speed"]
        self.serial_number = self.conf["serial_number"]
        self.system_under_measurement = self.conf["system_under_measurement"]
        # defaults = self.conf["defaults"]
        self.default_time_voltage = defaults["time_voltage"]
        self.default_time_current = defaults["time_current"]
        self.default_averaging = defaults["averaging"]
        self.ina_long_ids = list(self.conf["inas"].keys())

        # broker = self.conf["broker"]
        self.user = broker["user"]
        self.pw = broker["pw"]
        self.host = broker["host"]
        self.port = broker["port"]
        self.qos = broker["qos"]

        self.inas = self._get_inas()
        self.busses = self._get_busses()
        self.server = self._get_server()

    def _get_server(self):
        return [self.user, self.pw, self.host, self.port, self.qos]

    def _find_bus_id(self, ina_long_id):
        for bus, bus_inas in self.conf["busses"].items():
            bus_id = int(bus[-1])
            if bus_inas and ina_long_id in bus_inas:
                return bus_id
        return None
        # raise RuntimeError("Did not find YML INA ID {} in bus configuration!".format(ina_long_id))

    def _get_inas(self):
        found_inas = {}
        for ina_long_id, ina_dict in self.conf["inas"].items():
            a0 = ina_dict["address"]["a0"]
            a1 = ina_dict["address"]["a1"]
            a0_enum = LogicLevel[a0]
            a1_enum = LogicLevel[a1]
            address_obj = I2CAddress(a1_enum, a0_enum)
            time_voltage = ina_dict["time_voltage"] if "time_voltage" in ina_dict else self.default_time_voltage
            time_current = ina_dict["time_current"] if "time_current" in ina_dict else self.default_time_current
            averaging = ina_dict["averaging"] if "averaging" in ina_dict else self.default_averaging
            device_id = ina_dict["device_id"]
            r_shunt = ina_dict["r_shunt"]
            bus_id = self._find_bus_id(ina_long_id)
            if bus_id is not None:
                ina = INA(address_obj, device_id, bus_id, time_current, time_voltage, averaging, r_shunt,
                          ina_long_id)
                found_inas[ina_long_id] = ina
        return found_inas

    def _get_busses(self):
        busses = {}
        for bus, bus_inas in self.conf["busses"].items():
            bus_id = int(bus[-1])
            if bus_inas:
                if bus_id not in busses:
                    busses[bus_id] = []
                for ina_long_id in bus_inas:
                    if ina_long_id not in self.inas:
                        raise RuntimeError("Did not find YML INA ID {} that was "
                                           "configured for bus {} in INA configuration!".format(ina_long_id, bus))
                    busses[bus_id].append(self.inas[ina_long_id])
        bus_objs = [I2CBus(bus_inas) for bus_id, bus_inas in busses.items()]
        return bus_objs


class I2CBus:
    def __init__(self, inas: List[INA]):
        self.inas = inas
        self.bus_id = inas[0].bus_num

        addresses = {}
        for ina in inas:
            a = ina.address
            if a in addresses:
                raise RuntimeError("Two INAs with address {} found on bus {}: {} and {} - "
                                   "please make sure there are no "
                                   "duplicate addresses!".format(a, self.bus_id, ina.device_id, addresses[a].device_id))
            addresses[a] = ina


class TeensyBroker:
    def __init__(self, busses: List[I2CBus], port=None, serial_number=None, timeout=3, mqtt_name=None):
        self.mqtt_name = "teensy" if mqtt_name is None else mqtt_name
        self.timeout = timeout
        self.serial_number = serial_number
        if port is None:
            port = TeensyBroker.get_port(serial_number)
            if not port:
                raise RuntimeError("Did not find Teensy USB - please make sure"
                                   " you connected the Teensy device over USB with this machine!")
        self.port = port
        self.busses = busses
        print('ports:', self.port, flush=True)
        self.inas = [ina for bus in busses for ina in bus.inas]
        print('inas:', self.inas, flush=True)
        self.reconfigure(self.inas)
        self.last_print = None
        self.corrupted_packages = 0

    @staticmethod
    def get_port(serial_number=None):

        port = None
        baud = 9600
        ports = comports()
        for com in ports:
            if com.manufacturer is not None:
                if "teensy" in com.manufacturer.lower():
                    try:
                        if serial_number is None or str(com.serial_number) == str(serial_number):
                            port = com.device
                            board = Serial(port, baud)
                            board.close()
                            break
                    except:
                        pass
        return port

    def reconfigure(self, device_list):
        n_devices_to_configure = len(device_list)
        wait_time_out = 5
        print("Write first byte to init ReConfig Mode", flush=True)
        with open(self.port, "bw", ) as writeBuff:
            writeBuff.write(b'\xFF')

        print("Waiting to confirm ReConfig Mode", flush=True)
        with open(self.port, "rb") as readBuff:
            start_wait = time.time()
            signal_bytes = 0
            while signal_bytes < 20:
                b = readBuff.read(1)
                # print(b)
                if b == b'\xFF':
                    signal_bytes += 1
                else:
                    signal_bytes = 0
                if time.time() - start_wait > wait_time_out:
                    raise RuntimeError("Did not receive confirmation  after {} seconds (timeout)".format(wait_time_out))
        print("ReConfig Mode confirmed by Teensy", flush=True)

        with open(self.port, "bw", ) as writeBuff:
            for d in device_list:
                time.sleep(0.1)
                d.write_config_calib(writeBuff)
            writeBuff.write(b'\x00')
        print("Sent calibration & configuration to Teensy and sent final 0-Byte", flush=True)

        print("Waiting for number of online devices from Teensy", flush=True)
        with open(self.port, "rb", ) as readBuff:
            b = ''
            read_start = time.time()
            while len(b) == 0:
                b = readBuff.read(1)
                if time.time() - read_start > wait_time_out:
                    raise RuntimeError("Did not receive online status after {} seconds (timeout)".format(wait_time_out))
            n_found_devices = int.from_bytes(b, byteorder='big', signed=False)
            if n_found_devices == n_devices_to_configure:
                print("Feedback Teensy: Successfully configured", n_found_devices, "devices.", flush=True)

            # print("Receiving IDs of found INAs.")

            read_start = time.time()
            found_ids = []
            while True:
                b = readBuff.read(1)
                while len(b) == 0:
                    b = readBuff.read(1)
                    if time.time() - read_start > wait_time_out:
                        raise RuntimeError(
                            "Did not receive IDs of found INAs after {} seconds (timeout)".
                            format(wait_time_out))
                # id_str = b.to_bytes(1, byteorder='big', signed=False).decode("ascii")
                if int.from_bytes(b, byteorder='big', signed=False) == 0:
                    break
                id_str = b.decode("ascii")
                found_ids.append(id_str)
            # print() #falscher INAs: found_ids ist leer
            print("Found these devices:", flush=True)
            missing_inas = []
            for ina in self.inas:
                if ina.device_id in found_ids:
                    print(ina)
                else:
                    missing_inas.append(ina)

            if n_found_devices < n_devices_to_configure:
                print()
                print()
                for missing_ina in missing_inas:
                    print("MISSING: {}".format(missing_ina), flush=True)

                raise RuntimeError(
                    "Tried to configure {} INAs but only {} "
                    "INAs were found by Teensy on the I2C bus. Missing INAs are: {}".format(n_devices_to_configure,
                                                                                            n_found_devices,
                                                                                            missing_inas))

        time.sleep(0.2)
        print("\n\n")

    # FIXME err01: Payload decoding error. Description below. Also check out method de_mask().
    def read_usb_package_as_buffer(self, buffer, timeout):
        line = self.force_blocking_read_newline(buffer, timeout)
        if (self.last_print is None) or (time.time() - self.last_print > 10 * 60):
            print('line from buffer (406): ', line)
            self.last_print = time.time()
        ser_bytes = line[:-2]
        # print('ser_bytes:', ser_bytes)
        n_bytes = len(ser_bytes)
        if n_bytes == 8:
            payload = de_mask(ser_bytes[:7], ser_bytes[7])
        else:
            self.corrupted_packages += 1
            print("Corrupt package #{} received: {}".format(str(self.corrupted_packages), str(line)), flush=True)
            # sys.stdout.flush()
            return None, None, None, None
        try:
            device_id = payload[0].to_bytes(1, byteorder='big', signed=False).decode("ascii")
        except UnicodeDecodeError:
            return None, None, None, None  # FIXME err01

        sender_ina = None
        for ina in self.inas:
            # TODO after fix: remove elif-else for debugging
            if ina.device_id == device_id:
                sender_ina = ina
                break
            elif payload[0] == 10:
                print('payload[0] = 10:', device_id)  # debug err01
                break
        if sender_ina is None:
            # TODO: after fix, restore RuntimeError
            raise RuntimeError("Did not find INA ID {} in configured INAs.".format(device_id), device_id)
            return None, None, None, None  # FIXME err01

        power_bytes = payload[1:3]
        power_int: int = int.from_bytes(power_bytes, byteorder='big', signed=False)
        # controller_time = int.from_bytes(payload[4:8], byteorder='big', signed=False)
        time_3_7_little = int.from_bytes(payload[3:7], byteorder='little', signed=False)
        controller_time = time_3_7_little

        watts = sender_ina.get_power(power_int)
        return controller_time, device_id, watts, sender_ina

    def force_blocking_read_newline(self, buffer, timeout):
        read_start = time.time()
        line = []
        sleep_time = 1
        while len(line) == 0 or line[-1] != 10:
            line = [*line, *list(buffer.readline())]
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if len(line) < 10:
                # print("Line contains {} values. Skipping line. Time: {}. Line: {}".format(len(line), current_time, str(line)))
                time.sleep(1 / 10.0 ** 5)
            if time.time() - read_start > timeout:
                raise RuntimeError("Timeout during USB read. Time: {}. Line: {}. Sleep for {} seconds.".format(current_time, str(line), sleep_time))
        return line

    def read_continuous(self):
       #TODO re-connect catch exception
        with open(self.port, "br") as readBuff:
            self.clear_USB_buffer(readBuff)
            while True:
                controller_time, device_id, watts, ina = self.read_usb_package_as_buffer(readBuff,
                                                                                         timeout=self.timeout)
                if controller_time is None:
                    continue
                yield controller_time, ina, watts


    def benchmark(self, timeout=5, filter_id=None):
        with open(self.port, "br", ) as readBuff:
            start = time.time()
            last_readout = start
            received_list = []
            received_ids = set()
            mean_watts = []
            mean_micros = []

            self.clear_USB_buffer(readBuff)
            while time.time() - start < timeout:
                controller_time, device_id, watts, _ = self.read_usb_package_as_buffer(readBuff, timeout=timeout)
                if controller_time == None:
                    continue
                received_ids.add(device_id)
                if filter_id is None or device_id == filter_id:
                    received_list.append(watts)
                if time.time() - last_readout >= 1.0:
                    last_readout = time.time()
                    print("Received {} in {}s".format(len(received_list), 1))
                    if received_list:
                        rate_in_micros = 1000 * 1000 * 1 / len(received_list)
                        print("That equals 1 measurement every {} micros".format(rate_in_micros))
                        print("Mean power:", watts)
                        mean_watts.append(watts)
                        mean_micros.append(rate_in_micros)

                    print()
                    received_list.clear()
            print("Saw ids", sorted(received_ids))
            print("Mean micros =", np.mean(mean_micros))
            print("Mean power =", np.mean(mean_watts))

    def clear_USB_buffer(self, readBuff):
        _ = self.force_blocking_read_newline(readBuff, 2)


class MQTTSender:
    def __init__(self, server, teensy_broker: TeensyBroker, hostname, buffer_size=10000):
        self.user = server[0]
        self.pw = server[1]
        self.host = server[2]
        self.port = server[3]
        self.qos = server[4]

        self.measurement_buffer = []
        self.buffer_size = buffer_size
        self.teensy_broker = teensy_broker
        self.stop = False
        self.hostname = hostname # str(socket.gethostname())

        self.client = MQTTPublisher(user=self.user, pw=self.pw, host=self.host, port=self.port, qos=self.qos,
                                    hostname=self.hostname)

    def run(self):
        ina_list = {}
        micros_old = 0
        timeincrement = 0

        for micros, ina, watts in self.teensy_broker.read_continuous():
            max_micros = 4294967295
            time_since_last_measurement = micros - micros_old
            # print("{}|".format(time_since_last_measurement), end="")
            if time_since_last_measurement < -(10 ** 6):  # allow for different order within 1s
                timeincrement += 1
                # print("Time overflow after {} micros with old value {} and new value {}".format(max_micros, nano_old, nanos))
                # print("Time overflow happened {} times now".format(timeincrement))
                # print("Local time in s: {}", time.time())
                # print("Local time: {}".format(str(datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))))
            micros_old = micros
            micro_global_scale = micros + (
                    timeincrement * max_micros)  # OLD CONSTANT: 16777215 (16,777,215) USED (4,294,967,295) from https://www.pjrc.com/teensy/td_timing_millis.html
            self.publish_mqtt(ina_list, micro_global_scale, ina, watts)

            if self.stop:
                self.client.loop_stop()
                self.client.disconnect()
                print("run() self.stop")
                break

    def publish_mqtt(self, ina_list, nano_new, ina, watts):
        # measurement get added in a list of values in a dict of ina_name as key
        ina_name = str(ina).split()[-1]

        if ina_name not in ina_list:
            ina_list[ina_name] = []

        if not ina_list[ina_name]:
            ina_list[ina_name].append(nano_new)
            ina_list[ina_name].append(time.time())
        ina_list[ina_name].append(watts)

        # can publish max 20kb at once
        # after x values the list get published with ina_name as topic and set null
        if ina_list[ina_name].__len__() - 1 >= 800:
            ina_list[ina_name].append(nano_new)
            # print(ina_list[ina_name])
            self.client.publish(topic=ina_name, payload=ina_list[ina_name])
            ina_list[ina_name] = []


def main():
    parser = argparse.ArgumentParser(description='FAMLEM')
    parser.add_argument('-c', '--cfg', help='path for configuratioton file', default="config.yaml")
    parser.add_argument('-n', '--teensy-name', help='path for configuratioton file', required=True)
    args = vars(parser.parse_args())
    config_path = args["cfg"]
    teensy_name = args["teensy_name"]
    print("Using config", config_path)
    confs = YamlConfig(config_path)
    conf = confs.get_instance(teensy_name)
    busses = conf.busses
    login = conf.server
    serial_number = conf.serial_number
    system_under_measurement = conf.system_under_measurement

    while True:
        try:
            tb = TeensyBroker(busses, serial_number=serial_number)
            print(f"Found Teensy at port {tb.port}")
            device_id = None
            tb.benchmark(filter_id=device_id, timeout=3)

            mqtt = MQTTSender(server=login, teensy_broker=tb, buffer_size=100000, hostname=system_under_measurement)
            mqtt.client.client.loop_start()
            mqtt.run()
            mqtt.client.client.loop_stop()
        except RuntimeError:
            print("Lost connection to Teensy!")
            time.sleep(0.1)
            continue



# FIXME err01: Refine method?
# Possible error explanation (Johannes): Teensy probably uses '\n' for signaling the end of a package. '\n' found in the data portion of a package may lead to the package being cut short.
# The following byte will then be wrongly interpreted as the ID of the next sending INA even though it's truly a part of the previous package.
# This method was therefore designed to mask character '\n'.
def de_mask(payload, mask, mask_char='\n'):
    default_value = 128
    if mask != default_value:
        masked_val = int.from_bytes(mask_char.encode('ascii'), byteorder='big', signed=False)
        new_payload = []
        mask = list(BitArray(bytes=[mask], ))
        mask = mask[::-1]
        for i, (masked, payload_entry) in enumerate(zip(mask, payload)):
            if masked:
                new_payload.append(masked_val)
            else:
                new_payload.append(payload_entry)
    else:
        new_payload = payload
    return new_payload


def bits_fill_left(ba: bitarray):
    b = deepcopy(ba)
    b.reverse()
    b.fill()
    b.reverse()
    return b


if __name__ == "__main__":
    main()
