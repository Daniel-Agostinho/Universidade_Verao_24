import csv
import os
import time
import threading
import numpy as np
from bitalino import BITalino


class Device:
    def __init__(self, address, sampling_rate, channels, session_id):
        self.device = connect_bitalino(address)
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.session_id = session_id
        self.save_file = self.open_file()
        self.state = 0
        self.sample_number = 0
        self.session = False
        self.session_thread = None

    def start(self):
        self.session_thread = threading.Thread(target=self.start_acquisition)
        self.session_thread.start()

    def start_acquisition(self):
        self.session = True
        self.device.start(self.sampling_rate, self.channels)

        while self.session:
            data = self.get_data()
            self.save_data(data)

    def get_data(self):
        time_stamp = np.linspace(self.sample_number, self.sample_number + 1 - 1/self.sampling_rate, self.sampling_rate)
        states = np.zeros(self.sampling_rate).tolist()

        if self.state != 0:
            states[0] = self.state
            self.state = 0

        raw_data = self.device.read(self.sampling_rate)
        sensor1 = raw_data[:, -3]   # Resp
        sensor2 = raw_data[:, -2]   # ECG
        sensor3 = raw_data[:, -1]   # EDA

        data = np.vstack((time_stamp, sensor1, sensor2, sensor3, states)).transpose()

        self.sample_number += 1

        return data

    def save_data(self, data):
        with open(self.save_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def stop(self):
        self.session = False
        self.session_thread.join()
        self.device.stop()
        self.device.close()

    def open_file(self):
        file_fpath = os.path.join(
            "Data",
        )

        if not os.path.isdir(file_fpath):
            os.makedirs(file_fpath)

        file_name = os.path.join(
            file_fpath,
            f"session_{self.session_id}.csv",
        )

        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Resp", "ECG", "EDA", "Trigger"])

        return file_name


def connect_bitalino(address):
    connected = False
    device = None

    while not connected:
        try:
            print(f'Will connect to BITalino {address}')
            device = BITalino(address, timeout=5)
            connected = True

        except Exception:
            print("Failed to connect to device")
            print("Trying again in 2 seconds")
            time.sleep(2)

    print(f"Connected to device: {address}")

    return device


if __name__ == '__main__':
    pass
