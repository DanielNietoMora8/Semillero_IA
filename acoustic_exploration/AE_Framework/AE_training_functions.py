from scipy.io.wavfile import write
from six.moves import xrange
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchaudio.transforms as audio_transform
import torch.nn.functional as F
import wandb
from IPython.display import clear_output
from wandb import AlertLevel
import datetime
from datetime import timedelta


class TestModel:
    """
            Class to test convolutional autoencoder models.

    """

    """

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

    def __init__(self, model, iterator, num_views=8, device="cuda"):

        """
            :param model: Deep learning model.
            :type model:
            :param iterator: dataloader iterator.
            :type iterator:
            :param num_views: Specify the number of samples to visualize previously and after the reconstruction.
            :type num_views:
            :param device: Specify the device to do calculus.
            :type device:
            Todo:
                Check the args type.
            """

        self._model = model
        self._iterator = iterator
        self.num_views = num_views
        self.device = device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_waveform(self, waveform, directory=None):
        scaled = np.int16(waveform[0, 0] / np.max(np.abs(waveform[0, 0])) * 32767)
        write(directory + '.wav', 22050, scaled)

    def plot_waveform(self, waveform, n_rows=4):
        fig, axs = plt.subplots(n_rows, figsize=(10, 6), constrained_layout=True)
        for i in range(n_rows):
            axs[i].plot(waveform[i, 0])
        plt.show()

    def waveform_generator(self, spec, n_fft=1028, win_length=1028, audio_length=12, base_win=256):
        spec = spec.cdouble()
        spec = spec.to("cpu")
        # hop_length = int(np.round(base_win/win_length * 172.3))
        transformation = audio_transform.InverseSpectrogram(n_fft=n_fft, win_length=win_length)
        waveform = transformation(spec)
        waveform = waveform.cpu().detach().numpy()
        return waveform

    def plot_psd(self, waveform, n_wavs=1):
        for i in range(n_wavs):
            plt.psd(waveform[i][0])
            plt.xlabel("Frequency", fontsize=16)
            plt.ylabel("Power Spectral Density", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

    def plot_reconstructions(self, imgs_original, imgs_reconstruction):
        output = torch.cat((imgs_original[0:self.num_views], imgs_reconstruction[0:self.num_views]), 0)
        img_grid = make_grid(output, nrow=self.num_views, pad_value=20)
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.imshow(img_grid[1, :, :].cpu(), origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        plt.show()
        return fig

    def reconstruct(self):
        self._model.eval()
        (valid_originals, _, label, path) = next(self._iterator)
        valid_originals = torch.reshape(valid_originals, (valid_originals.shape[0] * valid_originals.shape[1]
                                                          * valid_originals.shape[2], valid_originals.shape[3],
                                                          valid_originals.shape[4]))
        valid_originals = torch.unsqueeze(valid_originals, 1)
        valid_originals = valid_originals.to(self.device)

        valid_encodings = self._model.encoder(valid_originals)

        valid_reconstructions = self._model.decoder(valid_encodings)

        valid_originals_nonorm = torch.expm1(valid_originals)
        valid_reconstructions_nonorm = torch.expm1(valid_reconstructions)

        BCE = F.mse_loss(valid_reconstructions, valid_originals)
        loss = BCE

        return valid_originals, valid_reconstructions, valid_encodings, label, loss, path

    def run(self, plot=True, wave_return=True, wave_plot=True, directory=None):
        wave_original = []
        wave_reconstruction = []
        originals, reconstructions, encodings, label, error = self.reconstruct()
        if plot:
            self.plot_reconstructions(originals, reconstructions)
        if wave_return:
            wave_original = self.waveform_generator(originals)
            wave_reconstruction = self.waveform_generator(reconstructions)
            if wave_plot:
                self.plot_waveform(wave_original, n_rows=4)
                self.plot_waveform(wave_reconstruction, n_rows=4)
            if directory != None:
                dir_ori = directory + "original_"
                dir_recon = directory + "reconstruction_"
                self.save_waveform(wave_original, dir_ori)
                self.save_waveform(wave_reconstruction, dir_recon)

        return originals, reconstructions, encodings, label, error


class TrainModel:

    def __init__(self, model):
        self._model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        print(self.device)

    def wandb_init(self, config, keys=["batch_size", "num_hiddens"]):
        try:
            run_name = str(config["architecture"]+"_")
            for key in keys:
                if key in config.keys():
                    run_name = run_name + key + "_" + str(config[key]) + "_"
                else:
                    run_name = run_name + str(key)

            wandb.login()
            wandb.finish()
            wandb.init(project="AE-Jaguas", config=config)
            wandb.run.name = run_name
            wandb.run.save()
            wandb.watch(self._model, F.mse_loss, log="all", log_freq=1)
            is_wandb_enable = True
        except Exception as e:
            print(e)
            is_wandb_enable = False

        return is_wandb_enable, run_name

    def wandb_logging(self, dict):
        for keys in dict:
            wandb.log({keys: dict[keys]})

    def fordward(self, training_loader, test_loader, config):
        iterator = iter(test_loader)
        wandb_enable, run_name = self.wandb_init(config)
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]

        # train_res_recon_error = []
        # train_res_perplexity = []
        logs = []
        # best_loss = 10000

        for epoch in range(config["num_epochs"]):
            iterator_train = iter(training_loader)
            for i in xrange(config["num_training_updates"]):
                self._model.train()
                try:
                    data, _, _, _ = next(iterator_train)
                except Exception as e:
                    print("error")
                    print(e)
                    logs.append(e)
                    continue

                data = torch.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
                data = torch.unsqueeze(data, 1)
                data = data.to(self.device)

                optimizer.zero_grad()
                data_recon = self._model(data)

                loss = F.mse_loss(data_recon, data)
                loss.backward()

                optimizer.step()
                print(
                    f'epoch: {epoch + 1} of {config["num_epochs"]} \t iteration: {(i + 1)} of {config["num_training_updates"]} \t loss: {np.round(loss.item(), 4)}')
                dict = {"loss": loss.item()}
                self.wandb_logging(dict)

                period = 200
                if (i + 1) % period == 0:
                    try:
                        test_ = TestModel(self._model, iterator, 8, device=torch.device("cuda"))
                        originals, reconstructions, encodings, labels, test_error, path = test_.reconstruct()
                        fig = test_.plot_reconstructions(originals, reconstructions)
                        images = wandb.Image(fig, caption=f"recon_error: {np.round(test_error.item(), 4)}")
                        self.wandb_logging({"examples": images, "step": (i + 1) // period})

                    except Exception as e:
                        print(f"error; {e}")
                        logs.append(e)
                        continue
                else:
                    pass

            scheduler.step()
            torch.cuda.empty_cache()
            time = datetime.datetime.now()
            torch.save(self._model.state_dict(),
                       f'temporal/models/model_{run_name}_day_{time.day}_hour_{time.hour}_epoch_{epoch+1}.pth')
            clear_output()
            print(optimizer.state_dict()["param_groups"][0]["lr"])

        wandb.finish()
        return self._model, logs, run_name
