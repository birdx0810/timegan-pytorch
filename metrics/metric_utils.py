# Necessary packages
import torch
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from metrics.general_rnn import GeneralRNN
from metrics.dataset import FeaturePredictionDataset, OneStepPredictionDataset

def rmse_error(y_true, y_pred):
    """User defined root mean squared error.

    Args:
    - y_true: true labels
    - y_pred: predictions

    Returns:
    - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred)**2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse

def reidentify_score(enlarge_label, pred_label):
    """Return the reidentification score.

    Args:
    - enlarge_label: 1 for train data, 0 for other data
    - pred_label: 1 for reidentified data, 0 for not reidentified data

    Returns:
    - accuracy: reidentification score
    """  
    accuracy = accuracy_score(enlarge_label, pred_label > 0.5)  
    return accuracy

def feature_prediction(train_data, test_data, index):
    """Use the other features to predict a certain feature.

    Args:
    - train_data (train_data, train_time): training time-series
    - test_data (test_data, test_data): testing time-series
    - index: feature index to be predicted

    Returns:
    - perf: average performance of feature predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters

    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 20
    args["batch_size"] = 128
    args["in_dim"] = dim-1
    args["h_dim"] = dim-1
    args["out_dim"] = 1
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = 100
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Output initialization
    perf = list()
  
    # For each index
    for idx in index:
        # Set training features and labels
        train_dataset = FeaturePredictionDataset(
            train_data, 
            train_time, 
            idx
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["batch_size"],
            shuffle=True
        )

        # Set testing features and labels
        test_dataset = FeaturePredictionDataset(
            test_data, 
            test_time,
            idx
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=no,
            shuffle=False
        )

        # Initialize model
        model = GeneralRNN(args)
        model.to(args["device"])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["learning_rate"]
        )

        logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
        for epoch in logger:
            running_loss = 0.0

            for train_x, train_t, train_y in train_dataloader:
                train_x = train_x.to(args["device"])
                train_y = train_y.to(args["device"])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                train_p = model(train_x, train_t)
                loss = criterion(train_p, train_y)
                # backward
                loss.backward()
                # optimize
                optimizer.step()

                running_loss += loss.item()

            logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

        
        # Evaluate the trained model
        with torch.no_grad():
            temp_perf = 0
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(args["device"])
                test_p = model(test_x, test_t).cpu().numpy()

                test_p = np.reshape(test_p, [-1])
                test_y = np.reshape(test_y.numpy(), [-1])
        
                temp_perf = rmse_error(test_y, test_p)
      
        perf.append(temp_perf)
    
    return perf
      
def one_step_ahead_prediction(train_data, test_data):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data
    
    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters
    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 20
    args["batch_size"] = 128
    args["in_dim"] = dim
    args["h_dim"] = dim
    args["out_dim"] = dim
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = 100 - 1   # only 99 is used for prediction
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(train_data, train_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(test_data, test_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=no,
        shuffle=True
    )
    # Initialize model
    model = GeneralRNN(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args["learning_rate"]
    )

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    with torch.no_grad():
        perf = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()

            test_p = np.reshape(test_p.numpy(), [-1])
            test_y = np.reshape(test_y.numpy(), [-1])

            perf += rmse_error(test_y, test_p)

    return perf