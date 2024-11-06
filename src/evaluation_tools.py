import os.path
import threading
import numpy as np
import pickle
import matplotlib.pyplot as plt
import neurokit2 as nk
from .record_tools import connect_bitalino, convert_units
from sklearn.tree import DecisionTreeClassifier


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
        self.classifier = load_model()
        self.session = False
        self.session_thread = None
        self.time = 0
        self.do_predict = False

        self.cached_data = {
            "Resp": np.zeros(sampling_rate * 8),
            "ECG": np.zeros(sampling_rate * 8),
            "EDA": np.zeros(sampling_rate * 8),
            "Lie": np.ones(sampling_rate * 8) * -1,
        }

        self.predict_data = None

    def start(self):
        self.session_thread = threading.Thread(target=self.start_acquisition)
        self.session_thread.start()

    def start_acquisition(self):
        self.session = True
        self.device.start(self.sampling_rate, self.channels)

        while self.session:
            self.get_data()
            self.lie_detect()

    def get_data(self):

        try:
            if self.do_predict and self.predict_data is None:
                pre_resp = self.cached_data["Resp"][-2 * self.sampling_rate:]
                pre_ecg = self.cached_data["ECG"][-2 * self.sampling_rate:]
                pre_eda = self.cached_data["EDA"][-2 * self.sampling_rate:]
                self.predict_data = np.vstack((pre_resp, pre_ecg, pre_eda)).transpose()

            raw_data = self.device.read(self.sampling_rate)

            sensor1 = raw_data[:, -3].tolist()   # Resp
            self.cache_data(sensor1, "Resp")

            sensor2 = raw_data[:, -2].tolist()   # ECG
            self.cache_data(sensor2, "ECG")

            sensor3 = convert_units(raw_data[:, -1]).tolist()   # EDA
            self.cache_data(sensor3, "EDA")

            if self.do_predict and self.predict_data is not None:
                new_data = np.vstack((sensor1, sensor2, sensor3)).transpose()
                self.predict_data = np.vstack((self.predict_data, new_data))

            self.time += 1

        except Exception:
            print("time out")

    def stop(self):
        self.session = False
        self.session_thread.join()
        self.device.stop()
        self.device.close()

    def cache_data(self, data, tag):
        old_data = self.cached_data[tag]
        new_data = np.append(old_data, data)
        self.cached_data[tag] = new_data[-self.sampling_rate * 8:]

    def lie_detect(self):

        if self.predict_data is None:
            self.cache_data(np.ones(self.sampling_rate) * -1, "Lie")
            return

        if self.predict_data.shape[0] == 10 * self.sampling_rate:

            # Do prediction
            value = self.predict()
            self.cache_data(np.ones(self.sampling_rate) * value, "Lie")

            # Reset for next prediction
            self.do_predict = False
            self.predict_data = None
            return

        self.cache_data(np.ones(self.sampling_rate) * -1, "Lie")

    def extract_features(self):

        resp_data = self.predict_data[:, 0]
        ecg_data = self.predict_data[:, 1]
        eda_data = self.predict_data[:, 2]

        our_features = []

        pre_signal = 2 * self.sampling_rate
        post_signal = 8 * self.sampling_rate
        trigger = np.zeros(10 * self.sampling_rate)
        trigger[pre_signal] = 1

        # ECG
        signals = nk.bio_process(ecg_data, sampling_rate=self.sampling_rate)
        events = nk.events_find(trigger, threshold=-0.0, event_conditions=[1], threshold_keep='above')
        epochs = nk.epochs_create(signals, events, sampling_rate=1000, epochs_start=-2.0, epochs_end=7.0)
        ecg_features = nk.ecg_analyze(epochs, sampling_rate=self.sampling_rate)

        x = ecg_features[['ECG_Rate_Baseline',
                            'ECG_Rate_Max', 'ECG_Rate_Min', 'ECG_Rate_Mean', 'ECG_Rate_SD',
                            'ECG_Rate_Max_Time', 'ECG_Rate_Min_Time', 'ECG_Rate_Trend_Linear',
                            'ECG_Rate_Trend_Quadratic', 'ECG_Rate_Trend_R2', 'ECG_Quality_Mean']]

        x['EDA_Mean'] = eda_data[-post_signal:].mean() / eda_data[:pre_signal].mean()
        x['EDA_SD'] = eda_data[-post_signal:].std() / eda_data[:pre_signal].std()
        x['EDA_Max'] = eda_data[-post_signal:].max() / eda_data[:pre_signal].mean()
        x['EDA_Min'] = eda_data[-post_signal:].min() / eda_data[:pre_signal].mean()

        # Add column to X with the Resp features
        x['Resp_Mean'] = resp_data[-post_signal:].mean() / resp_data[:pre_signal].mean()
        x['Resp_SD'] = resp_data[-post_signal:].std() / resp_data[:pre_signal].std()
        x['Resp_Max'] = resp_data[-post_signal:].max() / resp_data[:pre_signal].mean()
        x['Resp_Min'] = resp_data[-post_signal:].min() / resp_data[:pre_signal].mean()

        return x

    def predict(self):
        features = self.extract_features()
        predict = self.classifier.predict(features)
        return predict[0]


def live_plots(device):

    def key_press(event):
        print('press', event.key)
        if event.key == '1':
            if not device.do_predict:
                device.do_predict = True

    signal_layout = ["Resp", "ECG", "EDA", "Lie"]

    fig, axs, lines = build_plot(signal_layout)
    plt.show(block=False)
    live_plot = True
    fig.canvas.mpl_connect('key_press_event', key_press)
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
                ax.set_ylim(-1.5, 1.5)
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


def load_model():
    model_path = os.path.join(
        "src",
        "model",
        "model.pkl"
    )

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    return clf


if __name__ == '__main__':
    pass
