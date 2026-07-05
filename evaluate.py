import numpy as np
import torch 
from train import load_dataset, SolubilityNet

# Function to get the data used to validate in the model
def get_validation_data():
    X, y = load_dataset() # Loads in the data set with the function from train.py
    idx = np.random.RandomState(0).permutation(len(X)) #Uses the same random seed as train.py
    split = int(0.8 * len(X)) # Matches the 80-20 split in train.py
    va = idx[split:] # gets the same 20% validation held out
    return X[va], y[va] # Returns the portion of data used in validation

def main():
    Xva, yva = get_validation_data() # Loads in the validation data

    # Loads in the model into a variable
    model = SolubilityNet()
    model.load_state_dict(torch.load("model.pt"))
    model.eval() # Sets into evaluation mode, which turns off dropout

    with torch.no_grad(): #Turns off gradient bookeeping
        preds = model(torch.tensor(Xva)).numpy() # Converts validation fingerprints into a tensor, and stores predicted solubility for each

    rmse = np.sqrt(np.mean((preds - yva) ** 2)) # Calculate and stores the RMSE between predicted and actual solubilities

    # Creates a baseline prediction and error, just takes the average as the prediction
    baseline_pred = yva.mean()
    baseline_rmse = np.sqrt(np.mean((baseline_pred - yva) ** 2))
    
    ss_res = np.sum((yva - preds) ** 2) # Calculates sum of squares residual, comparing actual solubility to predicted solubility
    ss_tot = np.sum((yva - yva.mean()) **2) # Calculates the total sum of squares, that is how far apart the real values are
    r2 = 1 - ss_res / ss_tot # Calculates how much of the variation is explained by the model
    print(f"Validation molecules: {len(Xva)}")
    print(f"Model RMSE: {rmse:.3f} logS")
    print(f"Baseline RMSE: {baseline_rmse:.3f} logS -- This will guess the average")
    print(f"R^2: {r2:.3f} (1.0 = perfect, 0.0 = same as average)") # Prints out the R-squared, I got 0.713 in my run

if __name__ == "__main__":
    main()
