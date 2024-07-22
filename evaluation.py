import time
from src.evaluation_tools import Device, live_plots


def main():
    # Bitalino configuration
    bitalino_address = "00:21:08:35:15:17" # “/dev/tty.BITalino-XX-XX”
    bitalino_sampling_rate = 1000
    bitalino_channels = [0, 1, 2]

    device = Device(bitalino_address, bitalino_sampling_rate, bitalino_channels)

    # Starting session acquisition
    print("Start session acquisition")
    device.start()
    live_plots(device)
    device.stop()


if __name__ == '__main__':
    main()
