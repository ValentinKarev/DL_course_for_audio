import os
import glob
import torch
import torchaudio

class VCTKDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_len):
        self.root_dir = root_dir
        self.target_len = target_len

        # Собираем все wav файлы в корневой директории
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.wav")))
        if not self.files:
            raise ValueError(f"В директории {root_dir} не найдено .wav файлов")

        # Извлекаем ID диктора из имени файла
        # Предполагаемый формат: <speaker_id>_<...>.wav или <speaker_id>-<...>.wav
        # Если у вас другой формат, измените логику ниже
        self.file_speaker_ids = []
        for f in self.files:
            basename = os.path.basename(f)
            # Берём часть до расширения, затем до первого '_' или '-'
            name_without_ext = os.path.splitext(basename)[0]
            speaker_id = name_without_ext.split('_')[0].split('-')[0]
            self.file_speaker_ids.append(speaker_id)

        # Создаём маппинг ID диктора -> целочисленная метка класса
        unique_speakers = sorted(list(set(self.file_speaker_ids)))
        self.id2label = {spk: i for i, spk in enumerate(unique_speakers)}
        self.labels = [self.id2label[spk] for spk in self.file_speaker_ids]

        print(f"Найдено {len(self.files)} файлов от {len(unique_speakers)} дикторов.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(wav_path)

        # Приводим к моно (1, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Обрезаем или дополняем до нужной длины
        if waveform.shape[1] > self.target_len:
            max_start = waveform.shape[1] - self.target_len
            start_position = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start_position:start_position + self.target_len]
        elif waveform.shape[1] < self.target_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_len - waveform.shape[1]))
        # если == target_len, оставляем как есть

        # Убираем размер канала (1,) -> (T,)
        waveform = waveform.squeeze(0)

        # Нормализация по пиковому значению
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform, torch.tensor(label, dtype=torch.long)