import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet, DeepConvNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
###
import os
import json
from datetime import datetime


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs, outdir="."):
    # TODO plot training accuracy
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Training Accuracy')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'train_accuracy.png')); plt.close()

def plot_train_loss(train_loss_list, epochs, outdir="."):
    # TODO plot training loss
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'train_loss.png')); plt.close()

def plot_test_acc(test_acc_list, epochs, outdir="."):
    # TODO plot testing loss
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Testing Accuracy')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_accuracy.png')); plt.close()

def train(model, loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    ###
    os.makedirs('./weights', exist_ok=True)
    ###
    torch.save(best_wts, './weights/best.pt')
    ###
    print(f"\nTraining finished. Best test accuracy: {best_acc:.2f}%")
    ###
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    ###
    parser.add_argument("--model", type=str, default="eegnet",
                    choices=["eegnet", "deepconvnet"],
                    help="which model to run")
    parser.add_argument("--activation", type=str, default="elu",
                    choices=["elu", "leakyrelu", "relu", "gelu", "mish"],
                    help="activation function to use")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="dropout rate")
    parser.add_argument("--elu_alpha", type=float, default=1.0,
                        help="ELU alpha value (used only if activation=elu)")
    parser.add_argument("--temporal_filters", type=int, default=16,
                        help="number of temporal filters in first conv")
    parser.add_argument("--spatial_filters", type=int, default=32,
                        help="number of spatial filters in depthwise conv")
    parser.add_argument("--sweep", action="store_true",
                    help="run full grid of activations x dropouts x lrs")
    parser.add_argument("--out_root", type=str, default="runs",
                        help="root folder for all experiment outputs")
    ###
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO write EEGNet yourself
    if args.model == "eegnet":
        model = EEGNet(
            num_classes=2,
            dropout_rate=args.dropout,
            activation=args.activation,
            elu_alpha=args.elu_alpha,
            temporal_filters=args.temporal_filters,
            spatial_filters=args.spatial_filters
        )
    elif args.model == "deepconvnet":
        model = DeepConvNet(
            num_classes=2,
            C=2, T=750,
            activation=args.activation,
            elu_alpha=args.elu_alpha,
            dropout_rate=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)
    ###############################
    # Sweep Execution
    ###############################
    def fmt_float_for_path(x: float, digits=3):
        """將浮點數轉換為安全的路徑名稱字串"""
        if x >= 1e-2:
            s = f"{x:.{digits}f}".rstrip('0').rstrip('.')
            return s.replace('.', 'p')
        else:
            s = f"{x:.0e}"
            return s.replace('+0', '').replace('-0', '-')

    def make_tag(act, dr, lr):
        """資料夾名稱"""
        return f"act-{act}_d-{fmt_float_for_path(dr,2)}_lr-{fmt_float_for_path(lr,3)}"

    def build_model_from_args(args):
        return EEGNet(
            num_classes=2,
            dropout_rate=args.dropout,
            activation=args.activation,
            elu_alpha=args.elu_alpha,
            temporal_filters=args.temporal_filters,
            spatial_filters=args.spatial_filters
        )

    def run_one_experiment(args, act, dr, lr, outdir):
        args.activation = act
        args.dropout = dr
        args.lr = lr

        model = build_model_from_args(args).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

        train_acc, train_loss, test_acc = train(model, train_loader, criterion, optimizer, args)

        # 儲存圖與紀錄
        os.makedirs(outdir, exist_ok=True)
        plot_train_acc(train_acc, args.num_epochs, outdir)
        plot_train_loss(train_loss, args.num_epochs, outdir)
        plot_test_acc(test_acc, args.num_epochs, outdir)

        # 儲存每個 epoch 結果
        df = pd.DataFrame({
            "epoch": np.arange(1, args.num_epochs + 1),
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc
        })
        df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

        last_acc = float(test_acc[-1])
        best_acc = float(np.max(test_acc))
        best_epoch = int(np.argmax(test_acc) + 1)

        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump({
                "activation": act,
                "dropout": dr,
                "lr": lr,
                "elu_alpha": args.elu_alpha,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size
            }, f, indent=2)

        return last_acc, best_acc, best_epoch


    # -------------------------------
    # --sweep 組合設定
    # -------------------------------
    if args.sweep:
        activations = ["elu", "leakyrelu", "relu", "gelu", "mish"]
        dropouts = [0.20, 0.25, 0.30, 0.40]
        lrs = [0.01, 0.005, 0.003, 0.001]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(args.out_root, f"sweep_{timestamp}")
        os.makedirs(root, exist_ok=True)

        summary_rows = []
        for act in activations:
            for dr in dropouts:
                for lr in lrs:
                    tag = make_tag(act, dr, lr)
                    outdir = os.path.join(root, tag)
                    print(f"\n=== Running {tag} ===")
                    last_acc, best_acc, best_epoch = run_one_experiment(args, act, dr, lr, outdir)
                    summary_rows.append({
                        "activation": act,
                        "dropout": dr,
                        "lr": lr,
                        "last_acc": round(last_acc, 2),
                        "best_acc": round(best_acc, 2),
                        "best_epoch": best_epoch,
                        "folder": outdir
                    })

        summary = pd.DataFrame(summary_rows)
        summary.sort_values(by=["best_acc"], ascending=False, inplace=True)
        summary_path = os.path.join(root, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSweep done. Summary saved to {summary_path}")
        raise SystemExit(0)
    ###############################
    # End of sweep block
    ###############################


    train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, args)

    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)