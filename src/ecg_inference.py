import numpy as np
import tensorflow as tf

class ECGModel:
    def __init__(self, model_path: str = "models/swasthya_scan_ecg.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, ecg_signal_1d: np.ndarray) -> np.ndarray:
        # ecg_signal_1d shape: (187,)
        x = ecg_signal_1d.astype(np.float32).reshape(1, 187, 1)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])[0]
