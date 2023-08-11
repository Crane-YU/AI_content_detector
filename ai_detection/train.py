import os
import shutil

import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from data_process import run_data_preprocess
from model import ROBERTAClassifier
from utils import parse_arge, save_checkpoint, save_metrics

IS_SAGEMAKER = True


# load train and val datasets from a given folder
def load_datasets(args):
    # load dataset from the saved datapaths
    data_train_fp = os.path.join(args.datafolder, "train")
    data_val_fp = os.path.join(args.datafolder, "val")
    # if dataset files exist
    if os.path.exists(data_train_fp) and os.path.exists(data_val_fp):
        dataset_train = load_from_disk(data_train_fp)
        dataset_val = load_from_disk(data_val_fp)
    else:  # otherwise run the data_preprocess step
        run_data_preprocess(args)
        dataset_train = load_from_disk(data_train_fp)
        dataset_val = load_from_disk(data_val_fp)
    return dataset_train, dataset_val


# transfer the arrow type to torch type and keep the wanted columns
def set_dataloader(args, dataset_train, dataset_val):
    dataset_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    dataset_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataloader_train, dataloader_val


def train(
    args,
    model,
    device,
    dataloader_train,
    dataloader_val,
    optimizer,
    criterion,
    scheduler=None,
    validation_mode=True,
    checkpoint=True,
    metrics=True,
    save_dir="/opt/ml/model",
):
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float("Inf")

    print("===================================epoch total:", args.epochs)
    # Train loop
    for epoch in range(args.epochs):
        train_loss = 0.0
        num_samples = 0
        # turn the model to train mode
        model.train()
        for data in tqdm(dataloader_train):
            # gives batch data, extract input_ids, attention mask, and label
            # and make sure the data type is int
            data_train = data["input_ids"].long().to(device)
            mask = data["attention_mask"].long().to(device)
            label = data["label"].long().to(device)

            # add the number of data samples
            num_samples += data_train.shape[0]

            # run the model and compute the loss
            y_pred = model(input_ids=data_train, attention_mask=mask)
            loss = criterion(y_pred, label)

            # clear the gradient before backprop
            optimizer.zero_grad()
            # backprop, compute gradient
            loss.backward()
            # optimizer step
            optimizer.step()
            # use learning rate scheduler if given
            if scheduler is not None:
                scheduler.step()
            # Update train loss
            train_loss += loss.item()

        train_loss = train_loss / num_samples
        train_loss_list.append(train_loss)

        # Validation loop
        if validation_mode:
            valid_loss, acc_score = validation(model, device, dataloader_val, criterion)

            # Store train and validation loss history
            valid_loss = valid_loss / num_val_samples
            valid_loss_list.append(valid_loss)

            # whether to save the checkpoint
            if checkpoint:
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(os.path.join(args.output_path, "model.pkl"), model)

        # print summary
        if validation_mode:
            print(
                "Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.2f}".format(
                    epoch + 1, args.epochs, train_loss, valid_loss, acc_score
                )
            )
            # save the metrics
            if metrics:
                save_metrics(
                    os.path.join(args.output_path, "metric.pkl"),
                    train_loss_list,
                    valid_loss_list,
                    best_valid_loss,
                )
        else:
            print(
                "Epoch [{}/{}], Train Loss: {:.4f}".format(
                    epoch + 1, args.epochs, train_loss
                )
            )
            # save the metrics
            if metrics:
                save_metrics(
                    os.path.join(args.output_path, "metric.pkl"), train_loss_list
                )

    print("Training Done!")


def validation(model, device, dataloader_val, criterion):
    valid_loss = 0.0
    num_samples = 0
    label_list = []
    y_pred_list = []
    # turn the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader_val):
            data_val = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)
            label = data["label"].to(device)
            # add the number of data samples
            num_samples += data_val.shape[0]
            # compute model prediction
            y_pred = model(input_ids=data_val, attention_mask=mask)
            loss = criterion(y_pred, label)
            # compute the loss
            valid_loss += loss.item()
            # turn the predicted logits to labels
            y_pred_list.extend(torch.argmax(y_pred, dim=-1).tolist())
            label_list.extend(label.tolist())

    # return the validation loss and validation accuracy score
    valid_loss = valid_loss / num_samples
    acc_score = accuracy_score(label_list, y_pred_list)
    return valid_loss, acc_score


if __name__ == "__main__":
    # load the args
    args = parse_arge()

    if IS_SAGEMAKER:
        print(
            "==============Copying files in Sagemaker as initial model configs============="
        )
        source_folder = "/opt/ml/code/saved_model"
        destination_folder = "/opt/ml/model"

        for item in os.listdir(source_folder):
            s = os.path.join(source_folder, item)
            d = os.path.join(destination_folder, item)
            print(f"copying from {s} to {d}")
            try:
                if os.path.isdir(s):
                    shutil.copytree(
                        s, d, dirs_exist_ok=True
                    )  # dirs_exist_ok is available in Python 3.8+
                else:
                    shutil.copy2(s, d)
            except Exception as e:
                print(f"An error occurred: {e}")

    # fixed seed for repeatable implementation
    set_seed(args.seed)

    # load dataset from the saved datapaths
    dataset_train, dataset_val = load_datasets(args)

    # get the number of data samples
    num_train_samples = len(dataset_train)
    num_val_samples = len(dataset_val)

    # set dataload for train and validation
    dataloader_train, dataloader_val = set_dataloader(args, dataset_train, dataset_val)

    # define the model
    # model = ROBERTAClassifier(
    #     n_classes=2, dropout_rate=args.dropout_rate, model_path=args.saved_model_dir
    # )

    model = ROBERTAClassifier(n_classes=2, dropout_rate=args.dropout_rate)

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # put the model to the device
    model = model.to(device)

    # set training details: loss function, optimizer and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader_train),
        num_training_steps=len(dataloader_train) * args.epochs,
    )

    # train the model
    print("======================= Start Training ==============================")
    train(
        args,
        model,
        device,
        dataloader_train,
        dataloader_val,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        validation_mode=True,
        checkpoint=True,
        metrics=True,
        save_dir="/opt/ml/model",
    )
