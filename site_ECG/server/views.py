import io
from http.client import HTTPException

from django.shortcuts import render
from server.forms import PatientForm

import h5py

import torch
import torch.nn as nn
import os
import numpy as np

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, kernel_size=7):
        super(LSTMBinaryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=1)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)

        self.l_relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(hidden_size // 4, output_size)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        # Применяем сверточный слой
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len) -> (batch_size, seq_len, input_size)
        x = self.conv(x)  # (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out[:, -1, :])  # Берем только последний выход LSTM

        out = self.fc1(out)
        out = self.l_relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # Применяем сигмоиду для получения вероятности
        return out

# MODEL_PATH = "./weights/512_SGD_10_0.0001_0.9_0.pth"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "12_2048_SGD_10_0.0001_0.9_0.pth")
input_size = 12  # Один признак на каждый временной шаг
hidden_size = 50
output_size = 1  # Один выход для бинарной классификации
num_layers = 1

model = LSTMBinaryClassifier(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


def signal_transform_tensor_12(signals, N=0, max_len_signal=5000):
    """Транформирует сигналы в тензор.\n
    N - номер отведения в соответствии с массивом:\n
    ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']"""

    # Ограничиваем максимальную длину до единого значения для использования в батче.
    # Дописывать нули в конце для одной длины - сомнительная идея для LSTM.

    results_reshape = []

    for i in range(12):
        results_reshape.append(np.reshape([signal[i][0:max_len_signal] for signal in signals], (1, max_len_signal * len(signals))))

    result_reshape = np.concatenate(results_reshape)

    return torch.tensor(result_reshape, dtype=torch.float32).reshape((12, len(signals), max_len_signal)).permute((1,2,0))


# def process_h5_file(file):
#     try:
#         with h5py.File(io.BytesIO(file), "r") as f:
#             if "ecg" not in f:
#                 raise ValueError("Файл не содержит ключа 'ecg'")
#             ecg_data = f["ecg"][()]  # Загружаем данные
#
#             tensor_ecg = signal_transform_tensor_12(ecg_data)  # Применяем функцию
#
#             return tensor_ecg
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


def process_h5_file(file) -> torch.Tensor:
    try:
        with h5py.File(file, "r") as f:
            ecg_data = f["ecg"][()]
            print(ecg_data.shape)
            # if ecg_data.shape[0] != 12:
            #     return None
            return torch.tensor(ecg_data.T, dtype=torch.float32).unsqueeze(0)  # (1, N, 12)
    except:
        return None


def upload_data(request):
    if request.method == "POST":
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            patient = form.save()

            file = request.FILES["ekg_file"]
            signal_tensor = process_h5_file(file)


            if signal_tensor is None:
                patient.result = "Ошибка обработки файла"
            else:
                with torch.no_grad():
                    output = model(signal_tensor)
                    prediction = output.item()
                    print(prediction)
                    patient.result = "Болен" if prediction > 0.5 else "Здоров"
                    patient.result +=  f' ({1-prediction:.3f})'
            patient.save()
            return render(request, "result.html", {"result": patient.result})
    else:
        form = PatientForm()

    return render(request, "upload.html", {"form": form})