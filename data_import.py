import pandas as pd

# Importing the dataset from csv to pandas data frame
Fridge = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Fridge.csv")
Garage_Door = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Garage_Door.csv", dtype={"sphone_signal": "str"})
Gps_Tracker = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_GPS_Tracker.csv")
Modbus = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Modbus.csv")
Motion_Light = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Motion_Light.csv")
Thermostat = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Thermostat.csv")
Weather = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Weather.csv")

# Creating a new column to specify the device type for the joint data frame
Fridge["device_type"] = "fridge"
Garage_Door["device_type"] = "garage_door"
Gps_Tracker["device_type"] = "gps_tracker"
Modbus["device_type"] = "modbus"
Motion_Light["device_type"] = "motion_light"
Thermostat["device_type"] = "thermostat"
Weather["device_type"] = "weather"

# Joining all the data frames
IoT_data = pd.concat([Fridge, Garage_Door, Gps_Tracker, Modbus, Motion_Light, Thermostat, Weather], ignore_index=True)

#print(IoT_data.shape)
#print(IoT_data.columns)
#print(IoT_data.head())

# Clearing the data frame for unuseful data
IoT_data = IoT_data.drop(columns=["date", "time", "label"])

IoT_data.to_csv("IoT_data.csv", index=False)
#print(IoT_data.shape)
#print(IoT_data.columns)