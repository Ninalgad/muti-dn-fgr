# %%writefile /content/src/denoise.py
import click
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from preprocessing import *
from utils import DiceBCELoss
from dataset import DenosingDataset
from model import SimpNet
from ladp import LADP


def denoise_train_step(inp_batch, model, optimizer, device):
    inp, noise_tar = inp_batch['image'].to(device).float(), inp_batch['n_label'].to(device).float()
    v_tar = torch.unsqueeze(inp_batch["v_label"].to(device), 1)
    optimizer.zero_grad()

    noise_outputs, v_outputs = model(inp)

    # only compute loss for non-background elements
    mask = (inp > 0).float()
    loss_ = (noise_tar - noise_outputs) ** 2
    loss_ = torch.sum(loss_ * mask) / (torch.sum(mask) + 1)

    loss_ += DiceBCELoss(from_logits=True)(v_outputs, v_tar)

    loss_.backward()
    optimizer.step()
    return loss_.detach().cpu().numpy()


def train_denoise(model_name, df, noising_transform, steps_per_epoch,
                  num_epochs, learning_rate, batch_size,
                  debug=False):

    train_loader = DataLoader(
        DenosingDataset(df, steps_per_epoch * batch_size,
                        noising_transform=noising_transform),
        batch_size=batch_size, shuffle=True)

    model = SimpNet(3, pretrained=True, freeze_encoder=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    global_step = 0
    steps_per_epoch = len(train_loader)

    print('training')
    for epoch in range(num_epochs):
        print(f'Iter: {global_step}, Ep: {global_step / steps_per_epoch}')
        model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            loss = denoise_train_step(batch, model, optimizer, device)

            train_loss.append(loss)
            global_step += 1
            if debug:
                break
        print(epoch, np.mean(train_loss))

        torch.save({
            'model_state_dict': model.encoder.state_dict()
        }, f'{model_name}-pt.pt')


@click.command()
@click.option('--data-dir', type=str, help="Name of the model.")
@click.option('--model-name', type=str, help="Name of the model.")
@click.option('--split-seed', type=int, help="random seed for data split")
@click.option('--steps-per-epoch', type=int, default=-1, show_default=True, help="saving frequency")
@click.option('--num-epochs', type=int, default=10, show_default=True, help="number of training epochs")
@click.option('--learning-rate', type=float, default=1e-4, show_default=True, help="learning rate for Adam optimizer")
@click.option('--batch-size', type=int, default=16, show_default=True, help="number of images fed to the model at once")
@click.option('--debug', is_flag=True, show_default=True, default=False, help="run in debugging mode")
def run(data_dir, model_name, split_seed, steps_per_epoch,
        num_epochs, learning_rate, batch_size, debug):

    patients = get_patients_from_paths(data_dir)
    df_patient = create_patient_datafrmae(data_dir)

    training_patients, _ = train_test_split(patients, test_size=.1, random_state=split_seed)
    print(f"Number of patients used for training: {len(training_patients)}")

    df_train = df_patient[df_patient.patient.isin(training_patients)]

    if debug:
        print("Running in debugging mode")
        batch_size = 2
        df_train = df_train.iloc[:3]
        num_epochs = 1

    print(f"Training dataset size: {len(df_train)}")

    if steps_per_epoch <= 0:
        steps_per_epoch = int(len(df_train) // batch_size)

    train_denoise(model_name, df_train, LADP(), steps_per_epoch,
                  num_epochs, learning_rate, batch_size, debug)


if __name__ == '__main__':
    run()
