import torch
import torchaudio

import glob
import os

class AudioMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_len):
        self.root_dir = root_dir
        self.target_len = target_len
        self.files = []
        self.labels = []
        speaker_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
        for speaker_dir in speaker_dirs:
            for wav_path in glob.glob(os.path.join(speaker_dir, "*.wav")):
                basename = os.path.basename(wav_path)
                label_str = basename.split("_")[0]
                if label_str.isdigit():
                    self.files.append(wav_path)
                    self.labels.append(int(label_str))
        sorted_pairs = sorted(zip(self.labels, self.files))
        self.labels, self.files = [list(x) for x in zip(*sorted_pairs)]
        print(f"Найдено {len(self.files)} аудиофайлов")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < self.target_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_len - waveform.shape[1]))
        else:
            start_position = torch.randint(waveform.shape[1] - self.target_len)
            end = start_position + self.target_len
            waveform = waveform[start_position:end]
        waveform = waveform.squeeze()
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return waveform.float(), torch.tensor(self.labels[idx], dtype=torch.long)