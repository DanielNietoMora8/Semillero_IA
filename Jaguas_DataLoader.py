import torch
import torchaudio
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SoundscapeData(Dataset):

    """
    JaguasData is the main class used by the dataloader function allowing access to acoustic data of jaguas,
    and then training and testing a deep learning model. This version allows to define the audio length
    to return.
    """

    def __init__(self, root_path: str, audio_length: int, ext: str = "wav",
                 win_length: int = 255, original_length: int = 60):

        """
        This function is used to initialize the Dataloader, here path and root of files are defined.

        :param root_path: Files directory.
        :type root_path: str
        :param audio_length: Desired audio length partitions.
        :type audio_length: int
        :param ext: Audios format (e.g., .wav).
        :type ext: str
        :param ext: Audios extension (ex: .wav)
        :param win_length: Window length used to compute the fourier transform.
        :type win_length: int
        :param original_length: Duration in seconds of the original audios.
        :type original_length:
        """

        if 'google.colab' in str(get_ipython()):
            dir_root = "/content/drive/Shareddrives/"
        elif "zmqshell" in str(get_ipython()):
            dir_root = "/"
        else:
            dir_root = "G:/Unidades compartidas/"

        self.audio_length = audio_length
        self.original_length = original_length
        self.root_path = dir_root+root_path
        print(self.root_path)
        self.win_length = win_length
        self.folders = os.listdir(self.root_path)
        self.files = []

        for i in range(len(self.folders)):
            path_aux = "{}/{}".format(self.root_path, self.folders[i])
            self.files += list(Path(path_aux).rglob("*.{}".format(ext)))

    def __getitem__(self, index):

        """
        Function used to return audios and its respective spectrogram. Partitions of audios are made based on the
        users' parameterization. An additional axes for partitions is returned.

        :param index: index indicates the number of data to return.
        :returns:
            :spec: Spectrogram of the indexed audios.
            :type spec: torch.tensor
            :record: Array of indexed audios in monophonic format.
            :type record: numpy.array
            :label: Dictionary of labels including recorder, hour, minute and second keys.
            :type label: Dictionary
            :path_index: File directory.
            :type path index: String

        """
        if 'google.colab' in str(get_ipython()) or "zmqshell" in str(get_ipython()):
            delimiter = "/"
        else:
            delimiter = "\\"

        path_index = self.files[index]
        recorder = str(path_index).split(delimiter)[-2]
        recorder = int(recorder[1:3])
        hour = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][0:2])
        minute = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][2:4])
        second = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][4:6])
        label = {"recorder": np.repeat(recorder, self.original_length//self.audio_length),
                 "hour": np.repeat(hour, self.original_length//self.audio_length),
                 "minute": np.repeat(minute, self.original_length//self.audio_length),
                 "second": np.repeat(second, self.original_length//self.audio_length)}

        record, sr = torchaudio.load(path_index)
        resampling = 22050
        audio_len = self.audio_length * resampling
        record = torch.mean(record, dim=0, keepdim=True)
        record = torchaudio.transforms.Resample(sr, resampling)(record)
        missing_padding = resampling * self.original_length - record.shape[1]
        padding = torch.zeros([1, missing_padding])
        record = torch.cat((record, padding), axis=1)
        record = record[:, :audio_len * (record.shape[1] // audio_len)]
        record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
        win_length = self.win_length
        nfft = int(np.round(1*win_length))
        spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
                                                 window_fn=torch.hamming_window,
                                                 power=2,
                                                 normalized=False)(record)

        # spec = spec[0]
        spec = torch.log1p(spec)
        spec = torch.unsqueeze(spec, 0)
        # print(f"spec2: {spec.shape}")
        # spec = torch.unsqueeze(spec, dim=1)
        # db = F.AmplitudeToDB(top_db=80)
        # # print(record.shape)
        # spec = db(spec)
        # spec = torch.squeeze(spec, dim=1)
        path_index = str(path_index)
        return spec, record, label, path_index

    def __len__(self):

        """
        __len__ returns the len of the processed files

        :return: Number of processed files.
        """
        return len(self.files)
