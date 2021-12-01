# reference : https://github.com/peterliht/knowledge-distillation-pytorch

import os
import yaml
import json
import argparse
from tqdm import tqdm
from collections import namedtuple
from importlib import import_module
from sklearn.metrics import f1_score
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import loss
from src.model import Model
from src.dataloader import create_dataloader
from src.utils.torch_utils import save_model
from src.utils.common import get_label_counts, read_yaml

import teacher_models


def arg_parse():
    """
    parse arguments from a command
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    return args

def test(model, test_dataloader):
        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        # num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        num_classes = 6
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(device)
        model.eval()
        for batch, d in pbar:
            labels = d[0]['label'].long().to(device)
            data = d[0]['data']

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            running_loss += criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy

def train_kd(num_epochs, teacher, student, alpha, T, criterion, optimizer, scheduler, data_config, log_dir, scaler, device):
    train_dataloader, val_dataloader = create_dataloader(data_config)
    kld = nn.KLDivLoss(reduction='batchmean')
    best_test_acc = -1.0
    best_test_f1 = -1.0
    num_classes = 6
    label_list = [i for i in range(num_classes)]
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
    )
    for epoch in range(num_epochs):
        running_loss, running_hard_loss, running_kd_loss, correct, total = 0.0, 0.0, 0.0, 0, 0
        preds, gt = [], []
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        student.train()

        for batch, d in pbar:
            labels = d[0]['label'].long().to(device)
            data = d[0]['data']
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = student(data)
                    teacher_outputs = teacher(data)
            else:
                outputs = student(data)
                teacher_outputs = teacher(data)

            kd_loss = kld(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
            
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)

            hard_loss = criterion(outputs, labels)

            loss = kd_loss * alpha + hard_loss * (1 - alpha)

            optimizer.zero_grad()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()

            running_kd_loss += kd_loss.item()
            running_hard_loss += hard_loss.item()
            running_loss += loss.item()
            pbar.update()
            pbar.set_description(
                f"Train: [{epoch + 1:03d}] "
                f"KD: {(running_kd_loss / (batch + 1)):.3f}, "
                f"Hard: {(running_hard_loss / (batch + 1)):.3f}, "
                f"Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        pbar.close()

        _, test_f1, test_acc = test(student, val_dataloader)        
        if best_test_f1 > test_f1:
            continue
        best_test_acc = test_acc
        best_test_f1 = test_f1
        print(model_path)
        save_model(
            model=student,
            path=model_path,
            data=data,
            device=device,
        )
        print(f"Model saved. Current best test f1: {best_test_f1:.3f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_f1)


if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))
    os.makedirs(log_dir, exist_ok=True)

    num_epochs = cfgs.num_epochs

    # get data config
    data_config = read_yaml(cfg = cfgs.data)
    
    # create teacher model
    teacher_module = getattr(teacher_models, cfgs.teacher.name)
    teacher = teacher_module(**cfgs.teacher.args._asdict())
    teacher.load_state_dict(torch.load(cfgs.teacher_dir))
    teacher = teacher.to(device)
    teacher.eval()

    # create student model
    student_config = read_yaml(cfg = cfgs.student)
    student_instance = Model(student_config, verbose=True)
    student = student_instance.model.to(device)

    # create criterion
    if hasattr(loss, cfgs.criterion):
        criterion_module = getattr(loss, cfgs.criterion)
        criterion = criterion_module(
            samples_per_cls=get_label_counts(data_config["DATA_PATH"]) if data_config["DATASET"] == "TACO" else None,
            device=device)
    else:
        criterion_module = getattr(import_module("torch.nn"), cfgs.criterion)
        criterion = criterion_module(criterion)
    
    # create optimizer
    optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer.name)
    optimizer = optimizer_module(student.parameters(), **cfgs.optimizer.args._asdict())

    # create scheduler
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfgs.scheduler.name)
    scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())

    fp16 = cfgs.fp16
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    alpha = cfgs.alpha
    T = cfgs.T

    train_kd(num_epochs, teacher, student, alpha, T, criterion, optimizer, scheduler, data_config, log_dir, scaler, device)