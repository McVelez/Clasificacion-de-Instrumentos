from CustomDataset import MedleyDataset
from ModeloCNN import CNNNetwork
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

ANNOTATIONS_FILE = 'Medley_entrenamiento_Metadata.csv'
AUDIO_DIR = 'entrenamiento'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = .001

def create_data_loader(train_data, batch_size):
    train_dataLoader = DataLoader(train_data, batch_size=batch_size)
    return train_dataLoader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        
        # calculate loss 
        prediction = model(input)
        loss = loss_fn(prediction, target)
        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------------")
    print("Training complete")

if __name__ == "__main__":
    # Seleccionar GPU o CPU para entrenamiento
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    modeloPorUtilizar = input("QUE MODELO SE QUIERE ENTRENAR (default == mfcc): spectrogram | melspectrogram | mfcc \n")

    if modeloPorUtilizar == "melspectrogram":
        dimensionesCant = 2560
        transformada =  torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    elif modeloPorUtilizar == "spectrogram":
        dimensionesCant = 16896
        transformada = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)
    else:
        dimensionesCant = 4096
        transformada = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE)
    
    modelo = MedleyDataset(ANNOTATIONS_FILE, AUDIO_DIR, transformada, SAMPLE_RATE, NUM_SAMPLES, device, "MedleyS")
    train_dataloader = create_data_loader(modelo, BATCH_SIZE)

    # Build model y asignar a GPU o CPU identificado
    cnn = CNNNetwork(dimensionesCant).to(device)
    # Inicializar funcón de loss y optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    # Entrenar modelo
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Guardar modelo
    if modeloPorUtilizar == "melspectrogram":
        torch.save(cnn.state_dict(), "cnn_1.pth")
        print("Model trained and stored at cnn_1.pth")
    elif modeloPorUtilizar == "spectrogram":
        torch.save(cnn.state_dict(), "cnn_2.pth")
        print("Model trained and stored at cnn_2.pth")
    else:
        torch.save(cnn.state_dict(), "cnn_3.pth")
        print("Model trained and stored at cnn_3.pth")