# %%writefile /content/src/train.py
import click
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from utils import DiceBCELoss, FocalLoss
from dataset import BalancedTrainingDataset
from model import SimpNet
from eval import evaluate_metric
from preprocessing import get_patients_from_paths, create_patient_datafrmae


def combine_pn_batch(batch):
    return {
        "image": torch.cat([batch["image_pos"], batch["image_neg"]]),
        "s_label": torch.cat([batch["s_label_pos"], batch["s_label_neg"]]),
        "v_label": torch.cat([batch["v_label_pos"], batch["v_label_neg"]]),
    }


def segmentation_loss_func(outputs, tar):
    return DiceBCELoss(from_logits=True)(outputs, tar) + \
           3 * FocalLoss(from_logits=True)(outputs, tar)


def max_volume_loss_func(outputs, tar):
    return DiceBCELoss(from_logits=True)(outputs, tar)


def train_step(inp_batch, model, optimizer, device):
    inp, tar_seg = inp_batch['image'].to(device), inp_batch['s_label'].to(device).float()
    tar_fd = torch.unsqueeze(inp_batch["v_label"].to(device), 1)
    optimizer.zero_grad()

    prd_seg, prd_fd = model(inp)

    loss_ = segmentation_loss_func(prd_seg, tar_seg) + max_volume_loss_func(prd_fd, tar_fd)

    loss_.backward()
    optimizer.step()

    return loss_.detach().cpu().numpy()


def train_finetune(model_name, train_df, validation_df,
                   learning_rate, num_epochs, batch_size,
                   pt_model_name, pretrained,
                   val_epoch, debug):

    train_loader = DataLoader(BalancedTrainingDataset(train_df),
                              batch_size=batch_size // 2, shuffle=True)

    print('training')
    model = SimpNet(pretrained=pretrained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if os.path.isfile(f'{pt_model_name}-pt.pt'):
        print('loading weights')
        checkpoint = torch.load(f'{pt_model_name}-pt.pt', device)
        model.encoder.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint

    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_ep = 0
    global_step = 0
    best_val = -1
    best_thresh = -1

    steps_per_epoch = len(train_loader)
    if debug:
        num_epochs = 1

    for epoch in range(num_epochs):
        print(
            f'Iter: {global_step}, Ep: {global_step / steps_per_epoch}, '
            f'Current Best: {best_val} {best_thresh}, Best Ep: {best_ep}')
        model.train()
        train_loss = []

        for batch in tqdm(train_loader):
            batch = combine_pn_batch(batch)
            loss = train_step(batch, model, optimizer, device)

            train_loss.append(loss)
            global_step += 1
            if debug:
                break

        if epoch >= val_epoch:
            print(f"Current Best Val loss: {best_val} {best_thresh}, Best Ep: {best_ep}\n")
            with torch.no_grad():
                val, thresh = evaluate_metric(validation_df, model, device, debug=debug)

            print(f"Val: {val} {thresh}")
            if val > best_val:
                print(f"{epoch} New best Dice: {val} over {best_val}")
                best_val = val
                best_thresh = thresh
                best_ep = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val': best_val,
                    'best_thresh': best_thresh,
                }, f'{model_name}.pt')

    return best_val, best_thresh


@click.command()
@click.option('--data-dir', type=str, help="Name of the model.")
@click.option('--model-name', type=str, help="Name of the model.")
@click.option('--split-seed', type=int, help="random seed for data split")
@click.option('--learning-rate', type=float, default=5e-5, show_default=True, help="learning rate for Adam optimizer")
@click.option('--num-epochs', type=int, default=10, show_default=True, help="number of training epochs")
@click.option('--batch-size', type=int, default=16, show_default=True, help="number of images fed to the model at once")
@click.option('--pt-model-name', type=str, default=None, show_default=True, help="location of pre-pretrained model")
@click.option('--val-epoch', type=int, default=0, show_default=True, help="nuber of epochs without evaluation")
@click.option('--debug', is_flag=True, show_default=True, default=False, help="run in debugging mode")
def run(data_dir, model_name, split_seed, learning_rate, num_epochs, batch_size,
        pt_model_name, val_epoch, debug):

    patients = get_patients_from_paths(data_dir)
    df_patient = create_patient_datafrmae(data_dir)

    training_patients, val_patients = train_test_split(patients, test_size=.1, random_state=split_seed)
    print(f"Number of patients used for training: {len(training_patients)}, "
          f"Number of patients used for validation: {len(val_patients)}")

    df_train = df_patient[df_patient.patient.isin(training_patients)]
    df_val = df_patient[df_patient.patient.isin(val_patients)]

    if debug:
        print("Running in debugging mode")
        val_epoch = 0
        batch_size = 2

    print(f"Training dataset size: {len(df_train)}, "
          f"Validation dataset size: {len(df_val)}")

    s, t = train_finetune(model_name, df_train, df_val,
                          learning_rate, num_epochs, batch_size,
                          pt_model_name=pt_model_name, pretrained=True,
                          val_epoch=val_epoch, debug=debug)
    print(f"best score: {s} with threshold: {t}")


if __name__ == '__main__':
    run()
