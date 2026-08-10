# Listens to the MQTT topic, catches the vibration data, then evaluates it

import sys
import os
import json
import paho.mqtt.client as mqtt
from influxdb_client_3 import InfluxDBClient3, Point

# settings live next to the ESP32 host tools: credentials in the untracked
# private.py, everything non-secret in the tracked config.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'esp32', 'tools'))
from private import MQTT_BROKER, MQTT_TOPIC, INFLUX_URL, INFLUX_TOKEN, INFLUX_DATABASE, VIBRATION_THRESHOLD
from config import COLOR as C

client_v3 = InfluxDBClient3(host=INFLUX_URL, token=INFLUX_TOKEN, database=INFLUX_DATABASE)

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"{C['GREEN']}Edge Twin connected to MQTT Broker{C['RESET']}")
    client.subscribe(MQTT_TOPIC)
    print(f"Listening to topic: {MQTT_TOPIC}\n" + "-"*40)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    current_vibration = payload.get("rms_vibration")
    
    if current_vibration is None:
        print(f"{C['YELLOW']}[WARN]{C['RESET']} Key 'rms_vibration' missing from payload.")
        return

    # core logic of thesis heeere
    if current_vibration > VIBRATION_THRESHOLD:
        print(f"{C['RED']}[ALERT]{C['RESET']} Threshold Breached. Vibration: {current_vibration}")
        print(f"{C['YELLOW']}Offloading workload to the cloud.{C['RESET']}")
        influx_upload(current_vibration)
        # for later: package last 50 data points and send to cloud
    else:
        print(f"{C['GREEN']}[NORMAL]{C['RESET']} Edge Processing. Vibration: {current_vibration}")

def influx_upload(current_vibration):
    point = Point("vibration_analysis") \
        .tag("motor_id", "MTR-01") \
        .field("rms_vibration", current_vibration)
    
    client_v3.write(record=point)
    print(f"Data sent to InfluxDB: {current_vibration}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Edge Twin stopped")