import librosa
import soundfile
import torch

from waveglow_inference import TacotronSTFT, synthesize


def main():
    f = librosa.util.example_audio_file()
    sample_rate = 22050

    y, _ = librosa.load(f, sr=sample_rate)
    y = librosa.util.normalize(y)
    y = torch.from_numpy(y).unsqueeze(0)

    stft = TacotronSTFT()
    s = stft.mel_spectrogram(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform = synthesize(s, device=device)

    soundfile.write('example.wav',
                    waveform.squeeze(0).cpu().numpy(),
                    samplerate=sample_rate)


if __name__ == '__main__':
    main()
