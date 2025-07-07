import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for regression tasks.
    """
    def __init__(self, input_size: int, hidden: int = 64, dropout: float = 0.2) -> None:
        """
        Initialize the neural network.
        Args:
            input_size (int): Number of input features.
            hidden (int): Number of hidden units.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)

def train_nn(X, y, input_size, epochs=100, batch=32):
    model = SimpleNN(input_size)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1,1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return model, losses

def plot_learning_curve(model, X, y, fname='results/ml_learning_curve.png'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Train')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='Test')
    plt.xlabel('Training examples')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def model_selection(X, y):
    results = {}
    rf = RandomForestRegressor(n_estimators=100)
    rf_score = np.mean(cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error'))
    results['RandomForest'] = -rf_score
    if xgb_available:
        xgb = XGBRegressor(n_estimators=100)
        xgb_score = np.mean(cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_squared_error'))
        results['XGBoost'] = -xgb_score
    return results 