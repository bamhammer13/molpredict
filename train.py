import numpy as np
import pandas as pd
from features import smiles_to_fingerprint, FP_SIZE
import torch
# Imports PyTorch's neural network module as nn for convenience
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "data/esol.csv"
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"

# Function that loads in the molecules strings from the csv and converts them into fingerprints 
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X, y = [], []
    for smiles, target in zip(df[SMILES_COL], df[TARGET_COL]):
        try:
            X.append(smiles_to_fingerprint(smiles))
            y.append(float(target))
        except ValueError:
            continue # Skips the molecules RDKit can't parse
    X = np.stack(X)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

# Creates the SolubilityNet class, a child class of nn.Module, to serve as the basis for the neural network
class SolubilityNet(nn.Module):
    # Creates the initial model using Multi Layer Perceptron(MLP)
    # Default is a model with 512 - > 128 as the hidden dimensions and a dropout rate of 20%
    def __init__(self, in_dim=FP_SIZE, hidden_dims=(512, 128), dropout=0.2):
        # Calls the init of the parent class, nn.Module, to set up important variables such as the necessary registries  
        super().__init__()
        layers = []
        # The size of the last dimension starting with the input dimension
        prev = in_dim
        #Repeats to apply MLP to the hidden dimensions
        for d in hidden_dims:
            # Adds each hidden dimension layer to the layers list
            # Linear randomly generates weights and biases to matrix multiply prev numbers into d numbers, 
            # ReLu swaps negatives for 0
            # Dropout randomly zeroes a percentage of the values from over-memorizing our material
            layers += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)] 
            # Sets prev to the current dimension size for the next loop
            prev = d
        layers.append(nn.Linear(prev, 1)) # Applies the final MLP layer to get the singular value for the prediction
        # Bundles the layers into a network in self.net, which is part of SolubilityNet's .Module registry
        # Uses * to unpack layers from a list into singular values
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(0) # Sets the seed for repeatable randomness in the shuffle later
    X, y = load_dataset() # Loads in the dataset into X and y

    idx = np.random.RandomState(0).permutation(len(X)) # Shuffles the row order for fair distribution
    split = int(0.8 * len(X)) # Sets the split of training molecules to validation molecules at 80 to 20
    tr, va = idx[:split], idx[split:] # Splits the data into training and validation lists on the split

    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr]) # Creates the training input and target tensors
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va]) # Creates the validation input and target tensors

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True) # Divides data into shuffled batches of 32

    print("training molecules:", len(Xtr), " | validation molecultes: ", len(Xva))
    xb, yb = next(iter(loader))
    print("one batch - X:", xb.shape, "y", yb.shape)


# Test code for the parts of train.py
if __name__=="__main__":
    # Quick test of load dataset, showing what the data looks like after processing
    X, y = load_dataset()
    print("Loaded", len(X), "molecules")
    print("X shape:", X.shape, "| y shape:", y.shape)

    # Quick test running a fingerprint through an untrained dummy model and checking the shape of the output
    model = SolubilityNet()
    dummy = torch.tensor(X[:4]) # Uses the first 4 fingerprints
    out = model(dummy)
    print("Shape of model's output: ", out.shape) # Should be: "Shape of model's output:  torch.Size([4, 1])"

    main()