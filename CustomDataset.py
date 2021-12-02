import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class MedleyDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device, dataset):
        self.annotations = pd.read_csv(annotations_file) # archivo que contiene las etiquetas y metadata de los archivos de imagen preprocesados y segmentados
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.dataset = dataset

    # Para identificar la cantidad de archivos en el dataset
    def __len__(self):
        return len(self.annotations)

    # GetItem para acceder items del dataset
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal) # Reducir a un solo canal de audio
        signal = self._cut_if_neccesary(signal) # Num of samples es mayor que esperado
        signal = self._right_pad_if_necessary(signal) # Num of samples es menor que esperado 
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_neccesary(self, signal):
        # Tensor (1, num_samples) 
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples] # todos tendran un canal en relacion con hasta self.num_samples
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2, 16000) (channels, samples)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 3]}" # index para fila y columna 3 ya que es la columna de fold en el archivo csv
        
        if self.dataset == "MedleyS":
            if self.audio_dir == "entrenamiento":
                sub = "training"
            else:
                sub = "validacion"
            filename = f"Medley-solos-DB_{sub}-{self.annotations.iloc[index, 3]}_{self.annotations.iloc[index, 5]}.wav.wav"
        else:
            filename = self.annotations.iloc[index, 5]
        
        path = os.path.join(self.audio_dir, fold, filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2] # labels = [0 = guitarra electrica, 1 = flauta, 2 = piano, 3 = violin]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "Medley-solos-DB_metadata.csv"
    AUDIO_DIR = "audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = MedleyDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device,
                            "MedleyS"
                            )
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]