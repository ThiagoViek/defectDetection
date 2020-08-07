import os
import sys
import argparse 

import torch
from torch.utils.data import DataLoader
from torch import optim

ROOT_DIR = os.path.abspath("../")
PATH_CHECKPOINTS = os.path.abspath("checkpoints/")
PATH_DATA = os.path.join(ROOT_DIR, "data/")
sys.path.append(ROOT_DIR)

from model.oneShot import OSNet, ContrastLoss
from utils.dataset import Dataset

def train(model, optimizer, lossFn, trainLoader, valLoader, epochs, device):
    """Train network.
    model: model instance chosen.
    optimizer: optimizer chosen
    loss_fn: loss function defined.
    train_loader: train loader built with Dataset class
    val_loader: validation loader built with Dataset class
    epochs: how many epochs to run the model
    device: cpu or gpu
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    for epoch in range(epochs):
        trainingLoss = 0.0
        valLoss = 0.0

        model.train()
        for batch in trainLoader:
            optimizer.zero_grad()
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            y = batch["y"].to(device)
            
            D = model(img1, img2)
            loss = lossFn(D, y)
            loss.backward()
            optimizer.step()

            trainingLoss += loss.data.item()

        model.eval()
        for batch in valLoader:
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            y = batch["y"].to(device)

            D = model(img1, img2)
            loss = lossFn(D, y)
            
            valLoss += loss.data.item()
        
        trainingLoss /= len(trainLoader)
        valLoss /= len(valLoader)

        scheduler.step()

        print(f'\nEpoch: {epoch + 1} -  Mean Training Loss: {trainingLoss:.2f}')
        print(f'Epoch: {epoch + 1} -  Mean Validation Loss: {valLoss:.2f}\n')
    
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch Implementation for Defects Recognition on Steel Surfaces")
    parser.add_argument("--channels", 
                    help="Number of channels to consider in training (3 for RGB, 1 for Gray)", 
                    type=int,
                    required=True)
    parser.add_argument("--epochs", 
                    help="Epochs for training session", 
                    type=int,
                    default=100)
    parser.add_argument("--trainBatchSize", 
                    help="Batch size for training set", 
                    type=int,
                    default=32)
    parser.add_argument("--valBatchSize", 
                    help="Batch size for validation set", 
                    type=int,
                    default=4)
    parser.add_argument("--lr", 
                    help="Initial Learning rate", 
                    type=float,
                    default=5e-4)
    parser.add_argument("--margin", 
                    help="Margin to consider in ContrastiveLoss", 
                    type=float,
                    default=2)   
    parser.add_argument("--cpuThreads", 
                    help="Number of cpu Threads to use", 
                    type=int,
                    default=8) 
    parser.add_argument("--model", 
                    help="Load pre-trained model. Input must be the file name", 
                    type=str,
                    default=None)
    args = parser.parse_args()

    ########################### Setting up sesion ###########################
    print("Initializing Training...\n")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    net = OSNet(channels=args.channels)
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    epochs = args.epochs
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.999))
    lossFn = ContrastLoss(margin=args.margin)

    print(f"""Training Configurations:
        Image Channels: {args.channels}
        BatchSize Train: {trainBatchSize},
        BatchSize Validation: {valBatchSize},
        Epochs: {epochs},
        Initial Learning Rate: {args.lr},
        Loss Function: Contrastive Loss with Margin {args.margin}
        Optimizer: Adam
        Training Device: {device}
        \n""")

    if args.model is not None:
        modelPath = os.path.join(PATH_CHECKPOINTS, args.model)
        assert os.path.exists(modelPath), f"No model with name {args.model} found. Check for misspelling"
        
        print(f"Loading pre-trained Model at {checkpoints}\n")
        net.load_state_dict(torch.load(modelPath))

    # Dataset Configuration
    trainDir = os.path.join(ROOT_DIR, "data/train/")
    valDir = os.path.join(ROOT_DIR, "data/val/")

    trainData = Dataset(trainDir, args.channels)
    valData = Dataset(valDir, args.channels)

    trainLoader = DataLoader(trainData, batch_size=trainBatchSize, shuffle=True,
                            num_workers=args.cpuThreads, pin_memory=True)
    valLoader = DataLoader(valData, batch_size=valBatchSize, shuffle=True,
                            num_workers=args.cpuThreads, pin_memory=True)

    ############################# Training Model ##############################
    net.to(device)
    net = train(model = net, 
                optimizer = optimizer, 
                loss_fn = lossFn, 
                train_loader = trainLoader, 
                val_loader = valLoader, 
                epochs = epochs, 
                device = device)
    
    savingPath = os.path.join(PATH_CHECKPOINTS, "net_lastEpoch.pt")
    net.to('cpu')
    torch.save(net.state_dict(), savingPath)
    print("Training Session Finilized!")