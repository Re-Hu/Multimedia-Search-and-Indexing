import librosa
import numpy as np
import os

def calculate_and_save_mfcc(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

def preporcess_audio():
    audio_folder = 'Videos/Audios'
    mfcc_folder = 'Videos/MFCCs'

    if not os.path.exists(mfcc_folder):
        os.makedirs(mfcc_folder)

    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            audio_path = os.path.join(audio_folder, file)
            save_path = os.path.join(mfcc_folder, file + '.npy')
            calculate_and_save_mfcc(audio_path, save_path)


