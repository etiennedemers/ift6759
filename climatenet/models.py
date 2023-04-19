###########################################################################
# CGNet: A Light-weight Context Guided Network for Semantic Segmentation
# Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from time import time
from climatenet.utils.utils import Config
from os import path, makedirs
import json
import pathlib


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class CGNet:
    """
    The high-level CGNet class.
    This allows training and running CGNet without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : CGNetModule
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    """

    def __init__(self, config: Config = None, model_path: str = None, save_dir: str = 'results/',
                 transition_method='finetuning'):
        self.transition_method = transition_method
        self.config = config

        if model_path is not None:
            # Load model
            self.config = config
            if config is None:
                self.config = Config(path.join(model_path, 'config.json'))
            self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)),
                                       pretraining=self.config.pretraining, finetuning=self.config.pretraining).cuda()
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
            if self.config.pretraining:
                self.model_transition()

        elif config is not None:
            # Create new model
            self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)),
                                       pretraining=self.config.pretraining).cuda()

        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)
        self.save_dir = save_dir
        self.model_path = model_path

    def model_transition(self):
        if self.transition_method not in ['finetune', 'inception', 'classifier']:
            raise Exception('Invalid transition method!')

        if self.transition_method != 'finetune':
            for p in self.network.parameters():
                p.requires_grad = False

        if self.transition_method == 'inception':
            finalNet = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)),
                                   pretraining=0).cuda()
            finalNet.in_channels = 256
            self.network.classifier = nn.Sequential(ConvBNPReLU(256, 4, 3, 2),
                                                    Interpolate(size=(768, 1152), mode='bilinear'), finalNet).cuda()

        elif self.transition_method in ['classifier', 'finetune']:
            self.network.classifier = nn.Sequential(ConvBNPReLU(256, 128, 3, 2),
                                                    Conv(128, len(self.config.labels), 1, 1)).cuda()

    def pretrain(self):
        self.network.train()
        train_losses = []
        start = time()
        train_dataset = ClimateDatasetLabeled('data/pretrainSet', self.config)
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=1,
                            shuffle=True)
        for epoch in range(self.config.epochs['pretrain']):

            print(f'Epoch {epoch + 1}:')
            epoch_loader = tqdm(loader)

            train_quad_losses = []
            train_batch_sizes = []

            for features, labels in epoch_loader:
                # Push data on GPU and pass forward
                features = torch.tensor(features.values).cuda()
                labels = features.clone()

                outputs = self.network(features)

                # Pass backward
                loss = torch.mean((outputs - labels) ** 2)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_quad_losses.append(loss.item())
                train_batch_sizes.append(labels.shape[0])

            print('Epoch stats:')
            train_batch_sizes = np.array(train_batch_sizes)
            train_quad_losses = np.array(train_quad_losses)
            weighted_average_quad_losses = (train_batch_sizes * train_quad_losses).sum() / train_batch_sizes.sum()
            train_losses.append(weighted_average_quad_losses)
            print(f'Epoch Loss:{weighted_average_quad_losses}')

        end = time()
        hours = (end - start) // 3600
        minutes = ((end - start) - (hours * 3600)) / 60
        print(f'End of training. Total training time of {hours} hours and {minutes} minutes')
        print(f'Writing training logs to {self.save_dir}...')
        makedirs(self.save_dir, exist_ok=True)
        with open(path.join(self.save_dir, 'trainResults.json'), 'w') as f:
            f.write(json.dumps(
                {
                    "train_losses": train_losses,
                },
                indent=4,
            ))

    def train(self, train_data_path, curriculum: bool = False):
        '''Train the network on the given dataset for the given amount of epochs'''
        self.network.train()
        train_losses = []
        train_ious = []
        val_losses = []
        val_ious = []

        if curriculum:
            training_phases = ['simple', 'medium', 'hard']
        else:
            training_phases = ['union']

        val_dataset = ClimateDatasetLabeled(train_data_path + '/validationSet', self.config)

        start = time()
        for phase in training_phases:
            print(f'Starting {phase} training phase...')
            train_dataset = ClimateDatasetLabeled(train_data_path + phase + 'Set', self.config)
            collate = ClimateDatasetLabeled.collate
            loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, collate_fn=collate,
                                num_workers=1, shuffle=True)
            for epoch in range(self.config.epochs[phase]):

                print(f'Epoch {epoch + 1}:')
                epoch_loader = tqdm(loader)
                aggregate_cm = np.zeros((3, 3))

                train_jaccard_losses = []
                train_batch_sizes = []

                for features, labels in epoch_loader:
                    # Push data on GPU and pass forward
                    features = torch.tensor(features.values).cuda()
                    labels = torch.tensor(labels.values).cuda()

                    outputs = torch.softmax(self.network(features), 1)

                    # Update training CM
                    predictions = torch.max(outputs, 1)[1]
                    aggregate_cm += get_cm(predictions, labels, 3)

                    # Pass backward
                    loss = jaccard_loss(outputs, labels)
                    epoch_loader.set_description(f'Loss: {loss.item()}')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_jaccard_losses.append(loss.item())
                    train_batch_sizes.append(labels.shape[0])

                    train_jaccard_losses.append(loss.item())
                    train_batch_sizes.append(labels.shape[0])

                print('Epoch stats:')
                ious = get_iou_perClass(aggregate_cm)
                val_loss, val_iou = self.evaluate(val_dataset)
                val_losses.append(val_loss)
                val_ious.append(val_iou.mean())
                train_batch_sizes = np.array(train_batch_sizes)
                train_jaccard_losses = np.array(train_jaccard_losses)
                weighted_average_jaccard_losses = (
                                                          train_batch_sizes * train_jaccard_losses).sum() / train_batch_sizes.sum()
                train_losses.append(weighted_average_jaccard_losses)
                train_ious.append(ious.mean())
                print('IOUs: ', ious, ', mean: ', ious.mean())
                print('Val IOUs:', val_iou, ', mean:', val_iou.mean())

        end = time()
        hours = (end - start) // 3600
        minutes = ((end - start) - (hours * 3600)) / 60
        print(f'End of training. Total training time of {hours} hours and {minutes} minutes')
        print(f'Writing training logs to {self.save_dir}...')
        makedirs(self.save_dir, exist_ok=True)
        with open(path.join(self.save_dir, 'trainResults.json'), 'w') as f:
            f.write(json.dumps(
                {
                    "train_losses": train_losses,
                    "train_ious": train_ious,
                    "valid_losses": val_losses,
                    "valid_ious": val_ious
                },
                indent=4,
            ))

    def predict(self, dataset: ClimateDataset):
        """Make predictions for the given dataset and return them as xr.DataArray"""
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values).cuda()

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            coords = batch.coords
            del coords['variable']

            dims = [dim for dim in batch.dims if dim != "variable"]

            predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

        return xr.concat(predictions, dim='time')

    def evaluate(self, dataset: ClimateDatasetLabeled, verbose: bool = False):
        """Evaluate on a dataset and return statistics"""
        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=1)

        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3, 3))
        jaccard_losses = []
        batch_sizes = []

        for features, labels in epoch_loader:
            features = torch.tensor(features.values).cuda()
            labels = torch.tensor(labels.values).cuda()

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            aggregate_cm += get_cm(predictions, labels, 3)

            jaccard_losses.append(jaccard_loss(outputs, labels.cpu()).item())
            batch_sizes.append(labels.shape[0])

        ious = get_iou_perClass(aggregate_cm)
        batch_sizes = np.array(batch_sizes)
        jaccard_losses = np.array(jaccard_losses)
        weighted_average_jaccard_losses = (batch_sizes * jaccard_losses).sum() / batch_sizes.sum()

        if verbose:
            print('Evaluation stats:')
            print(aggregate_cm)
            print('IOUs: ', ious, ', mean: ', ious.mean())

        return weighted_average_jaccard_losses, ious

    def get_cm_for_each_file(self, dataset: ClimateDatasetLabeled):

        assert self.config.pred_batch_size == 1, "can only get the ious for each file if pred_batch_size=1"

        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=1)

        epoch_loader = tqdm(loader)
        cms = []

        for features, labels in epoch_loader:
            features = torch.tensor(features.values).cuda()
            labels = torch.tensor(labels.values).cuda()

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]

            cms.append(get_cm(predictions, labels, 3))

        return np.stack(cms)

    def save_model(self, save_path: str):
        """
        Save model weights and config to a directory.
        """
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

        # save weights and config
        self.config.save(path.join(save_path, 'config.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights.pth'))

    def load_model(self, model_path: str):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))


class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """

    def __init__(self, classes=19, channels=4, M=3, N=21, dropout_flag=False, pretraining=False, finetuning=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.pretraining = pretraining
        self.finetuning = finetuning
        self.in_channels = channels

        self.level1_0 = ConvBNPReLU(self.in_channels, 32, 3, 2)  # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + self.in_channels)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 + self.in_channels, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
        self.bn_prelu_2 = BNPReLU(128 + self.in_channels)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + self.in_channels, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag and not pretraining:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1))
        elif not dropout_flag and not pretraining:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, channels, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        if not self.finetuning and self.pretraining:
            # Gaussian Noise
            # eps = 0.1
            # noise = eps * torch.randn(input.size()).cuda()
            # input += noise
            # Dropout
            input = F.dropout(input)
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',
                            align_corners=False)  # Upsample score map, factor=8
        return out
