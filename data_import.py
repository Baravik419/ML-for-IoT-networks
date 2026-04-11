import pandas as pd
import glob

# Importing the IoT dataset from csv to pandas data frame
Fridge = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Fridge.csv")
Garage_Door = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Garage_Door.csv", dtype={"sphone_signal": "str"})
Gps_Tracker = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_GPS_Tracker.csv")
Modbus = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Modbus.csv")
Motion_Light = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Motion_Light.csv")
Thermostat = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Thermostat.csv")
Weather = pd.read_csv("ToN_IoT dataset (IoT only)/Processed_IoT_dataset/IoT_Weather.csv")

# Creating a new column to specify the device type for the joint IoT data frame
Fridge["device_type"] = "fridge"
Garage_Door["device_type"] = "garage_door"
Gps_Tracker["device_type"] = "gps_tracker"
Modbus["device_type"] = "modbus"
Motion_Light["device_type"] = "motion_light"
Thermostat["device_type"] = "thermostat"
Weather["device_type"] = "weather"

# Joining all the IoT data frames
IoT_data = pd.concat([Fridge, Garage_Door, Gps_Tracker, Modbus, Motion_Light, Thermostat, Weather], ignore_index=True)

#print(IoT_data.shape)
#print(IoT_data.columns)
#print(IoT_data.head())

# Converting date and time to timestamp

IoT_data["ts"] = (
    pd.to_datetime(
    IoT_data["date"].str.strip() + " " + IoT_data["time"].str.strip(),
    format="%d-%b-%y %H:%M:%S"
    ).astype("int64") // 10**6
)

# Importing the network dataset from 23 csv files to pandas data frame
Network_data = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob("ToN_IoT dataset (Network only)/Network_dataset_*.csv")], ignore_index=True)

#print(Network_data.shape)
#print(Network_data.columns)
#print(Network_data.head())

# Merging the IoT and Network data frames
keys = ["ts", "label", "type"]

IoT_network_data = Network_data.merge(
    IoT_data[keys].drop_duplicates(),
    on=keys,
    how="inner"
)

#print(IoT_network_data.shape)
#print(IoT_network_data.columns)
#print(IoT_network_data.head())

IoT_network_data.to_csv("IoT_network_data.csv", index=False)
