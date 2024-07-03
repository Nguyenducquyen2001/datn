import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchinfo import summary
from torchvision import datasets, models, transforms

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_model = "gui/model"

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(7, hidden_dim, num_layers = 2, bidirectional=True, batch_first=True)        
        self.linear1 = nn.Linear(hidden_dim, 32)
        self.linear2 = nn.Linear(32, output_dim)

    def forward(self, x):
        output_lstm, (hidden_lstm, cell_lstm) = self.lstm(x)
        last_hidden = hidden_lstm[-1,:,:]
        linear1 = self.linear1(last_hidden)
        linear2 = self.linear2(linear1)
        return linear2
    
model_acc_sg = LSTMModel(hidden_dim=64, output_dim=2)
model_audio = models.resnet18()

model_acc_sg.load_state_dict(torch.load(save_model + '/acc_sg_model.pt'))

in_features = model_audio.fc.in_features
model_audio.fc = nn.Linear(in_features, 2)
model_audio.load_state_dict(torch.load(save_model + '/best_audio.pt'))

model_acc_sg.to(device)
model_acc_sg.eval()

model_audio.to(device)
model_audio.eval()

# Processing acc_sg data
import joblib
scaler = joblib.load(save_model + "/scaler.pkl")

def preprocess_data(df):
    df = df.values
    fx = df[:, 0]
    fx_av = np.convolve(fx, np.ones(10) / 10, mode="same")
    fy = df[:, 1]
    fy_av = np.convolve(fy, np.ones(10) / 10, mode="same")
    ax = df[:, 2]
    ay = df[:, 3]
    az = df[:, 4]
    f_av = np.sqrt((fx_av) ** 2 + (fy_av) ** 2)
    a_av = np.sqrt((ax) ** 2 + (ay) ** 2 + (az) ** 2)
    return np.c_[fx_av, fy_av, f_av, ax, ay, az, a_av]

def result_from_acc_sg_model(df_acc_sg):
    df_acc_sg_pr = preprocess_data(df_acc_sg)
    df_acc_sg_scaler = scaler.transform(df_acc_sg_pr.reshape(-1, 7)).reshape(-1, 160, 7)
    df_acc_sg_tensor = torch.tensor(df_acc_sg_scaler, dtype=torch.float32).to(device)
    result = model_acc_sg(df_acc_sg_tensor)
    probality = F.softmax(result, dim=-1)
    return probality


# Processin audio data
data_path = "./gui/data/audio_image/audio.jpg"
IMG_SIZE = 224

def preprocess_audio_data(df):
    df = df.values.reshape(-1)
    df = df / 3277
    return df

def spectrogram_image(data):
    stft = librosa.stft(data)
    stft_db = librosa.amplitude_to_db(abs(stft))
    plt.figure(figsize=(10,10))
    librosa.display.specshow(stft_db, sr=44000)
    
image_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 , 0.5 , 0.5], [0.5 , 0.5 , 0.5])])

def result_from_audio_model(df_audio):
    df_audio_pr = preprocess_audio_data(df_audio)
    spectrogram_image(df_audio_pr)
    plt.savefig(data_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open(data_path)
    img_transform = image_transforms(img).reshape(-1, 3, IMG_SIZE, IMG_SIZE).to(device)
    result = model_audio(img_transform)
    probality = F.softmax(result, dim=-1)
    return probality

def ensemble_model(df_acc_sg, df_audio):
    proba_acc_sg = result_from_acc_sg_model(df_acc_sg)
    proba_audio = result_from_audio_model(df_audio)
    proba_final = proba_acc_sg * 0.7 + proba_audio * 0.3
    proba_final = proba_final.to("cpu")
    proba0 = proba_final[0][0].item()
    return proba0, torch.argmax(proba_final)
