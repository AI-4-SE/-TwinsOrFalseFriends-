import json
import math
import time
from abc import ABC

import paho.mqtt.client as mqtt
from yaml import load


class MQTTClient(ABC):

    def __init__(self, user, pw, host, port, qos, hostname, clean_session=True):

        self.host = host
        self.port = port
        self.hostname = hostname
        self.client_id = self.hostname + '_session'
        self.client = mqtt.Client(client_id=self.client_id, clean_session=clean_session)
        self.client.username_pw_set(user, pw)

        self.client.on_subscribe = self.on_subscribe

        self.is_connected = False
        self.client.on_connect = self.on_connect
        self.client.connect(self.host, self.port, keepalive=60)

        self.qos = qos
        self.test_connection()

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("subscribed to: " + str(client))

    def test_connection(self):
        start = time.time()
        while not self.is_connected:
            self.client.loop()
            time.sleep(0.1)
            if time.time() - start > 2:
                print('Connection to broker failed.')
                exit(1)

    def hostname_to_topic(self, hostname):
        topics = []
        # todo add curie
        if any('publisher' or 'fetcher' or 'curie' for _ in hostname):
            topics.append(hostname[:5])
            topics.append(hostname[5:])
        elif 'edison' in hostname:
            topics.append(hostname[:6])
            topics.append(hostname[6:])
        elif 'hopper' in hostname:
            topics.append('notImplemented')
            topics.append('hopper')
        else:
            topics.append('notImplemented')
            topics.append(hostname)
        return topics

    def on_connect(self, client, userdata, flags, rc):  # The callback for when the client connects to the broker
        # if rc == 0: print("connected to: " + self.host)
        self.is_connected = True
        # print("Connected with result code {0}".format(str(rc)))  # Print result of connection attempt


class MQTTFetcher(MQTTClient):
    def __init__(self, user, pw, host, port, qos, hostname, clean_session, is_ffem=False):
        super().__init__(user, pw, host, port, qos, hostname, clean_session=clean_session)
        self.is_ffem = is_ffem
        self.client.on_message = self.on_message

        self.client.on_disconnect = self.on_disconnect
        self.max_loop_time = 5

        self.client.loop_start()

        self.topics = []
        self.measurements_ffem = {}
        self.measurements = []

    # differenz between normal/ ffem
    def on_message(self, client, userdata, msg):  # The callback for when a PUBLISH message is received from the server.
        # print("Message received-> " + msg.topic + " " + str(msg.payload, 'UTF-8'))  # Print a received msg
        if self.is_ffem:
            if msg.topic not in self.measurements_ffem:
                self.measurements_ffem[msg.topic] = []
            self.measurements_ffem[msg.topic].append(json.loads(str(msg.payload, 'UTF-8')))
        else:
            self.measurements.append(json.loads(str(msg.payload, 'UTF-8')))

    def on_disconnect(self, client, userdata, rc):
        # print("disconnecting reason  " + str(rc))
        self.is_connected = False

    # differenz between normal/ ffem
    def get_topic(self):
        # reads ina's from yaml and adds them to topiclist
        if self.is_ffem:
            with open("default-config.yaml") as yaml:
                config = load(yaml)
                for bus in config["busses"]:
                    if config["busses"][bus] is not None:
                        for ina in config["busses"][bus]:
                            self.topics.append('/'.join(self.hostname_to_topic(self.hostname)) + "/" + ina)
                            self.topics.append('/'.join(self.hostname_to_topic(self.hostname)) + "/" + ina)
                            self.topics.append('/'.join(self.hostname_to_topic(self.hostname)) + "/" + ina)
            return self.topics
        # default
        return '/'.join(self.hostname_to_topic(self.hostname) + ['#'])

    # differenz between normal/ ffem (get_topic)
    def subscribe(self):
        for topic in self.get_topic():
            print(self.client.subscribe(topic, qos=self.qos))
            print(topic)
        # print('subscribed to topic:', self.get_topic())

    # differenz between normal/ ffem (get_topic)
    def unsubscribe(self):
        for topic in self.get_topic():
            self.client.unsubscribe(topic=topic)
        # print('unsubscribed to topic:', self.get_topic())

    def disconnect(self):
        self.client.disconnect()

    def loop(self):
        self.client.loop(self.max_loop_time + 0.1)

    def get_measurements(self):
        if self.measurements:
            return self.measurements
        return self.measurements_ffem

    def get_max_loop_time(self):
        return self.max_loop_time


class MQTTPublisher(MQTTClient):
    def __init__(self, user, pw, host, port, qos, hostname):
        super().__init__(user, pw, host, port, qos, hostname=hostname)
        # self.connect()
        self.client.loop_start()

    def publish_test(self, num_messages):
        print("TEST: sending", num_messages, "messages")
        t0 = time.time()
        for i in range(num_messages):
            self.client.publish("/testing/value1", str(math.sin(0.1 * i)), qos=self.qos)
            self.client.publish("/testing/value2", str(math.cos(0.1 * i)), qos=self.qos)
            time.sleep(0.001)
        print('elapsed time', time.time() - t0)
        print('values', num_messages)

    def publish_m(self, topics, m):
        self.client.publish('/'.join(topics), m, qos=self.qos)

    def begin(self):
        pass

    def take_samples(self, samples):
        # print('publish ', len(samples), 'samples')
        # new_samples = []
        for (t, pdu_dict) in samples:
            for hostname, measurements in pdu_dict.items():
                for metric, value in measurements.items():
                    row = t, hostname, metric, value
                    topics = self.hostname_to_topic(hostname)
                    m_json = {"t": t, "val": value}
                    self.publish_m(topics + [metric], json.dumps(m_json))

                    # m_json = {t: value}
                    # self.publish_m(topics+[metric], json.dumps(m_json))

    def publish(self, topic, payload):
        topic_p = self.hostname_to_topic(self.hostname)
        t = '/'.join([topic_p[0], topic_p[1], topic])
        json_payload = json.dumps(payload)
        self.client.publish(topic=t, payload=json_payload, qos=self.qos)

    def stop(self):
        pass
