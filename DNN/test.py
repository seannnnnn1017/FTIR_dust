import paho.mqtt.client as mqtt

# Define the MQTT broker address and port
broker_address = "test.mosquitto.org"
port = 1883

# Define the topic to subscribe to
topic = "op23756778"

# Define the callback function for when a message is received
def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()}")

# Create an MQTT client and set the callback function
client = mqtt.Client()
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, port=port)

# Subscribe to the topic
client.subscribe(topic)

# Loop to keep the script running and handle incoming messages
client.loop_forever()