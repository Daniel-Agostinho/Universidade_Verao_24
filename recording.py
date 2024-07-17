from src.record_tools import Device
import keyboard


def main():

    # Bitalino configuration
    bitalino_address = "00:21:08:35:15:17"
    bitalino_sampling_rate = 1000
    bitalino_channels = [0, 1, 2]

    session_id = input("Session ID: ")
    device = Device(bitalino_address, bitalino_sampling_rate, bitalino_channels, session_id)

    # Starting session acquisition
    print("Start session acquisition")
    device.start()

    while True:
        key = keyboard.read_key()
        device.state = key

        if key == "q":
            device.stop()
            break

    print("Ending session")


if __name__ == '__main__':
    main()
