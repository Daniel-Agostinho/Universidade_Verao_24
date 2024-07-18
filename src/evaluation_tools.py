import threading
import numpy as np
import matplotlib.pyplot as plt
from .record_tools import connect_bitalino


SIGNAL_COLOR = {
    "Resp": "b",
    "ECG": "r",
    "EDA": "y",
    "Lie": "k",
}


class Device:
    def __init__(self, address, sampling_rate, channels):
        self.device = connect_bitalino(address)
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.classifier = None
        self.session = False
        self.session_thread = None
        self.time = 0

        self.cached_data = {
            "Resp": np.zeros(sampling_rate * 8),
            "ECG": np.zeros(sampling_rate * 8),
            "EDA": np.zeros(sampling_rate * 8),
            "Lie": np.zeros(sampling_rate * 8),
        }

    def start(self):
        self.session_thread = threading.Thread(target=self.start_acquisition)
        self.session_thread.start()

    def start_acquisition(self):
        self.session = True
        self.device.start(self.sampling_rate, self.channels)

        while self.session:
            self.get_data()

    def get_data(self):
        raw_data = self.device.read(self.sampling_rate)

        sensor1 = raw_data[:, -3].tolist()   # Resp
        self.cache_data(sensor1, "Resp")

        sensor2 = raw_data[:, -2].tolist()   # ECG
        self.cache_data(sensor2, "ECG")

        sensor3 = raw_data[:, -1].tolist()   # EDA
        self.cache_data(sensor3, "EDA")

        # TODO Classifier predict
        # Place holder test
        if self.time == 6 or self.time == 15:
            self.cache_data(np.ones(self.sampling_rate), "Lie")
        else:
            self.cache_data(np.zeros(self.sampling_rate), "Lie")

        self.time += 1

    def stop(self):
        self.session = False
        self.session_thread.join()
        self.device.stop()
        self.device.close()

    def cache_data(self, data, tag):
        old_data = self.cached_data[tag]
        new_data = np.append(old_data, data)
        self.cached_data[tag] = new_data[-self.sampling_rate * 8:]

    def predict(self):
        pass


def live_plots(device):

    signal_layout = ["Resp", "ECG", "EDA", "Lie"]

    fig, axs, lines = build_plot(signal_layout)
    plt.show(block=False)
    live_plot = True

    plt.ion()

    while live_plot:
        time = device.time
        fs = device.sampling_rate
        x = build_x(time, fs)

        signals = device.cached_data

        for idx, signal in enumerate(signal_layout):
            ax = axs[idx]
            data = signals[signal]
            lines[idx].set_data(x, data)
            ax.set_xlim(time - 8, time)

            if signal == "Lie":
                ax.set_ylim(-0.5, 1.5)
            else:
                ax.set_ylim(data.min() - 5, data.max() + 5)

            ax.draw_artist(lines[idx])
            fig.canvas.blit(ax.bbox)

        fig.tight_layout()
        fig.canvas.flush_events()

        if not plt.fignum_exists(1):
            live_plot = False

    plt.ioff()


def build_plot(signals):

    fig, axs = plt.subplots(4, 1, figsize=(19.20, 10.80))
    lines = []

    for idx, signal in enumerate(signals):
        line, = axs[idx].plot([], color=SIGNAL_COLOR[signal])
        axs[idx].set_title(signal)
        axs[idx].set_xlabel("Tempo (s)")
        axs[idx].set_ylabel("Voltagem (mV)")
        lines.append(line)

    return fig, axs, lines


def build_x(time, fs):
    x = np.linspace(time - 8, time, fs * 8)
    return x


if __name__ == '__main__':
    pass
