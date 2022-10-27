import argparse
import pathlib
import warnings
import os
from collections import OrderedDict, Counter

import random
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import transforms
from pytorch_metric_learning import miners, losses, testers, distances, samplers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import sigver.datasets.util as util
from sigver.featurelearning.data import TransformDataset
import sigver.featurelearning.models as models


def train(base_model: torch.nn.Module,
          classification_layer: torch.nn.Module,
          forg_layer: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          train_set: torch.utils.data.TensorDataset,
          val_set: torch.utils.data.TensorDataset,
          device: torch.device,
          args: Any,
          logdir: Optional[pathlib.Path]):
    """ Trains a contrastive model

    Parameters
    ----------
    base_model: torch.nn.Module
        The model architecture that "extract features" from signatures
    classification_layer: torch.nn.Module
        The classification layer (from features to predictions of which user
        wrote the signature)
    forg_layer: torch.nn.Module
        The forgery prediction layer (Only used in SigNet).
    train_loader: torch.utils.data.DataLoader
        Iterable that loads the training set (x, y) tuples
    val_loader: torch.utils.data.DataLoader
        Iterable that loads the validation set (x, y) tuples
    device: torch.device
        The device (CPU or GPU) to use for training
    args: Namespace
        Extra arguments for training: epochs, lr, lr_decay, lr_decay_times, momentum, weight_decay
    logdir: str
        Where to save the model and training curves

    Returns
    -------
    Dict (str -> tensors)
        The trained weights

    """

    # Collect all parameters that need to be optimizer
    parameters = base_model.parameters()

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                              nesterov=True, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             args.epochs // args.lr_decay_times,
                                             args.lr_decay)

    best_acc = 0
    best_epoch = 0
    best_params = get_parameters(base_model, classification_layer, forg_layer)


    #Loss type (NTXent or Triplet)
    if args.loss_type == 'ntxent':
        print('Training with NT-Xent loss using temperature given by',args.temperature)
        distance = distances.CosineSimilarity()
        loss_func = losses.NTXentLoss(temperature=args.temperature,distance=distance)
        mining_func = None
    else:
        print('Training with Triplet loss using margin given by',args.margin)
        distance = distances.LpDistance()
        loss_func = losses.TripletMarginLoss(margin = args.margin,distance=distance)
        mining_func = miners.TripletMarginMiner(margin = args.margin, type_of_triplets = 'semihard')
    
    #Metrics
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
   
    for epoch in range(args.epochs):
        # Train one epoch; evaluate on validation
        train_loss, mined_triplets = train_epoch(train_loader, base_model, classification_layer, forg_layer,
                    epoch, optimizer, lr_scheduler, device, loss_func, mining_func, args)

        val_acc, val_loss = test(train_set, val_set, val_loader, epoch, base_model, loss_func, accuracy_calculator, device)

        #Save the best model only on improvement (early stopping)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_params = get_parameters(base_model, classification_layer, forg_layer)
            if logdir is not None:
                torch.save(best_params, logdir / 'model_best.pth')

        if logdir is not None:
            current_params = get_parameters(base_model, classification_layer, forg_layer)
            torch.save(current_params, logdir / 'model_last.pth')

    print('Best epoch was the number:', best_epoch)
    return best_params, best_epoch


def copy_to_cpu(weights: Dict[str, Any]):
    return OrderedDict([(k, v.cpu()) for k, v in weights.items()])


def get_parameters(base_model, classification_layer, forg_layer):
    best_params = (copy_to_cpu(base_model.state_dict()),
                   copy_to_cpu(classification_layer.state_dict()),
                   copy_to_cpu(forg_layer.state_dict()))
    return best_params


def train_epoch(train_loader: torch.utils.data.DataLoader,
                base_model: torch.nn.Module,
                classification_layer: torch.nn.Module,
                forg_layer: torch.nn.Module,
                epoch: int,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                loss_func: losses.BaseMetricLossFunction,
                mining_func: miners.TripletMarginMiner,
                args: Any) -> float:
    """ Trains the network for one epoch

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            Iterable that loads the training set (x, y) tuples
        base_model: torch.nn.Module
            The model architecture that "extract features" from signatures
        classification_layer: torch.nn.Module
            The classification layer (Only used in SigNet)
        forg_layer: torch.nn.Module
            The forgery prediction layer (Only used in SigNet).
        epoch: int
            The current epoch (used for reporting)
        optimizer: torch.optim.Optimizer
            The optimizer (already initialized)
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler
        device: torch.device
            The device (CPU or GPU) to use for training
        args: Namespace
            Extra arguments used for training:

        Returns
        -------
        None
        """

    step = 0
    n_steps = len(train_loader)
    losses = 0
    mined_triplets = 0
    base_model.train()
    for batch in train_loader:
        x, y = batch[0], batch[1]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        # Forward propagation
        optimizer.zero_grad()
        features = base_model(x)
        
        if mining_func is not None:
            indices_tuple = mining_func(features, y)
            num_triplets = mining_func.num_triplets
        else:
            indices_tuple = None
            num_triplets = 0

        loss = loss_func(features, y, indices_tuple)

        # Back propagation
        loss.backward()
        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)

        # Update weights
        optimizer.step()

        step += 1
        losses += float(loss)
        mined_triplets += num_triplets

    train_loss = losses/n_steps
    avg_mined_triplets = mined_triplets
    
    print("Epoch {}: Average training loss per batch = {:.4f}".format(epoch, train_loss))
    if 'Triplet' in str(type(loss_func)):
        print("Epoch {}: Total of mined triplets in epoch = {:.2f}".format(epoch, avg_mined_triplets))
    lr_scheduler.step()

    return train_loss, avg_mined_triplets


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set: torch.utils.data.TensorDataset,
         val_set: torch.utils.data.TensorDataset,
         val_loader: torch.utils.data.DataLoader,
         epoch: int,
         base_model: torch.nn.Module,
         loss_func: losses.BaseMetricLossFunction,
         accuracy_calculator: AccuracyCalculator,
         device: torch.device) -> float:
    """ Test the model in a validation/test set

    Parameters
    ----------
    val_loader: torch.utils.data.DataLoader
        Iterable that loads the validation set (x, y) tuples
    base_model: torch.nn.Module
        The model architecture that "extract features" from signatures
    classification_layer: torch.nn.Module
        The classification layer (from features to predictions of which user
        wrote the signature)
    device: torch.device
        The device (CPU or GPU) to use for training
    Returns
    -------
    float, float
        The valication accuracy and validation loss

    """

    #Computing validation loss
    n_steps = len(val_loader)
    losses = 0
    for batch in val_loader:
        x, y = batch[0], batch[1]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        with torch.no_grad():
            features = base_model(x)
            loss = loss_func(features, y)
            losses += float(loss)

    val_loss = losses/n_steps
    print("Epoch {}: Average validation loss per batch = {:.4f}".format(epoch, val_loss))

    #Clear gpu before computing accuracy
    torch.cuda.empty_cache()
    #Computing validation accuracy
    print("Computing validation accuracy...")
    train_features, train_labels = get_all_embeddings(train_set, base_model)
    val_features, val_labels = get_all_embeddings(val_set, base_model)

    accuracies = accuracy_calculator.get_accuracy(val_features, 
                                                train_features,
                                                np.squeeze(val_labels),
                                                np.squeeze(train_labels),
                                                False)

    val_acc = accuracies["precision_at_1"]
    print("\nValidation set accuracy (Precision@1) = {}".format(val_acc))
   
    return val_acc, val_loss

def main(args):
    # Setup logging
    logdir = pathlib.Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir()

    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Loading Data')

    x, y, yforg, usermapping, filenames = util.load_dataset(args.dataset_path)
    data = util.get_subset((x, y, yforg), subset=range(*args.users))
    data = util.remove_forgeries(data, forg_idx=2)

    train_loader, val_loader, train_set, val_set = setup_data_loaders(data, args.batch_size, args.input_size)

    print('Initializing Model')

    n_classes = len(np.unique(data[1]))

    base_model = models.available_models['signet']().to(device)
    
    classification_layer = nn.Module()  # Stub module with no parameters    
    forg_layer = nn.Module()  # Stub module with no parameters

    if args.task == 'multi':
        print('Weights are being loaded from:', args.logdir + '/model_best.pth')
        base_model_params, classification_params, forg_params = torch.load(args.logdir + '/model_best.pth')
        base_model.load_state_dict(base_model_params)
        #Classification_params and forg_params are used only in SigNet

    train(base_model, classification_layer, forg_layer, train_loader, val_loader,
          train_set, val_set, device, args, logdir)


def setup_data_loaders(data, batch_size, input_size):

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[1])

    data = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y))
    train_size = int(0.9 * len(data))
    sizes = (train_size, len(data) - train_size)
    train_set, test_set = random_split(data, sizes)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
    ])
    train_set = TransformDataset(train_set, train_transforms)
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    test_set = TransformDataset(test_set, val_transforms)

    sampler = samplers.MPerClassSampler(train_set.targets, m=2, batch_size=batch_size, length_before_new_iter=2*len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, train_set, test_set


def build_arguments():

    warnings.filterwarnings("ignore")

    argparser = argparse.ArgumentParser('Train Second task')
    argparser.add_argument('--dataset-path', help='Path containing a numpy file with images and labels')
    argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(150, 220))
    argparser.add_argument('--users', nargs=2, type=int, default=(5000, 7000))

    argparser.add_argument('--batch-size', help='Batch size', type=int, default=256)
    argparser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    argparser.add_argument('--lr-decay', help='learning rate decay (multiplier)', default=0.1, type=float)
    argparser.add_argument('--lr-decay-times', help='number of times learning rate decays', default=3, type=float)
    argparser.add_argument('--momentum', help='momentum', default=0.90, type=float)
    argparser.add_argument('--weight-decay', help='Weight Decay', default=1e-4, type=float)
    argparser.add_argument('--epochs', help='Number of epochs', default=60, type=int)
    argparser.add_argument('--task', help='Single-task or multi-task', choices=('multi','single'), default='multi', type=str)

    argparser.add_argument('--seed', default=42, type=int)
    argparser.add_argument('--loss-type', help='Loss type', choices=('triplet','ntxent'), default='triplet', type=str)
    argparser.add_argument('--margin', help='margin', default=0.1, type=float)
    argparser.add_argument('--temperature', help='temperature', default=0.01, type=float)

    argparser.add_argument('--gpu-idx', default=0, type=int)
    argparser.add_argument('--logdir', help='logdir', required=True)

    #arguments = argparser.parse_args()
    arguments, unknown = argparser.parse_known_args()
    print(arguments)

    main(arguments)


if __name__ == '__main__':
    build_arguments()
