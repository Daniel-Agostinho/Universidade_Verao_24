import time

from src.record_tools import Device
import keyboard

TIME_TRIAL = 24
N_QUESTIONS = 18


def main():

    # Bitalino configuration
    bitalino_address = "00:21:08:35:15:17" # “/dev/tty.BITalino-XX-XX”
    bitalino_sampling_rate = 1000
    bitalino_channels = [0, 1, 2]

    session_id = input("Session ID: ")
    device = Device(bitalino_address, bitalino_sampling_rate, bitalino_channels, session_id)

    # Starting session acquisition
    print("Start session acquisition\n")
    device.start()
    time.sleep(TIME_TRIAL)

    question = 1
    while question <= N_QUESTIONS:
        device.state = "p"
        print(f"Pergunta {question}!")
        print("Espera resposta...")

        key = keyboard.read_key()   # Block
        print(f"Resposta : {key}\n")
        device.state = key
        time.sleep(TIME_TRIAL)
        question += 1

    device.state = "q"
    device.stop()

    print("Ending session")


if __name__ == '__main__':
    main()
