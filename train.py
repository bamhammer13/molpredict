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

    # print statements to verify split and batching worked
    print("training molecules:", len(Xtr), " | validation molecules: ", len(Xva))
    xb, yb = next(iter(loader))
    print("one batch - X:", xb.shape, "y", yb.shape)

    model = SolubilityNet() # Creates an instance of SolubilityNet
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Creates an Adam optimizer
    loss_fn = nn.MSELoss() # Creates a loss function

    # Runs 50 passes over the data
    for epoch in range(1, 51):
        model.train() # Sets the model into training mode, enabling dropout
        for xb, yb in loader: # Loops through the batches
            optimizer.zero_grad() # Clears old gradients
            loss = loss_fn(model(xb), yb) # Produces a prediction and calculates how wrong it was
            loss.backward() # Computes gradients for the weights
            optimizer.step() # Shifts weights to decrease loss

        # Shows progresss every 10 epochs
        if epoch % 10 == 0:
            model.eval() # Sets the model into evaluation mode, disabling dropout

            #Calculates Root Mean Squared Error, uses no_grad to skip gradient tracking since we're only lookingw
            with torch.no_grad(): 
                tr_rmse = torch.sqrt(loss_fn(model(Xtr),  ytr)).item()
                va_rmse = torch.sqrt(loss_fn(model(Xva),  yva)).item()
            print(f"epoch {epoch:3d}    train RMSE {tr_rmse:.3f}    val RMSE {va_rmse:.3f}")
    torch.save(model.state_dict(), "model.pt") # Saves the model to model.pt
    print("Saved trained model to model.pt")

# Test code for the parts of train.py
if __name__=="__main__":
    main()