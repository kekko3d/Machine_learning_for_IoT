from time import time
from time import sleep
import paho.mqtt.client as mqtt
import json
import uuid
import psutil as psu

def main():

    client = MQTT_Client()
    client.connect()

    while(True):

        timestamp_ms = int(time() * 1000)
        battery = psu.sensors_battery()

        message = {
            "mac_address" : hex(uuid.getnode()),
            "timestamp" : timestamp_ms,
            "battery_level" : battery.percent,
            "power_plugged" : bool(battery.power_plugged)
        }

        message_json = json.dumps(message, indent=4)
        client.publish('XXXXXX', message_json)
        sleep(1)


class MQTT_Client():

    def __init__(self):
        self.client = mqtt.Client()

    def connect(self):
        self.client.connect("mqtt.eclipseprojects.io", 1883, 60)
 
    def publish(self, topic, message):
        self.client.publish(topic, message)


if __name__ == '__main__':
    main()
