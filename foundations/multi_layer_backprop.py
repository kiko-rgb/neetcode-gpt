import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x  = np.array(x,  dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        z1 = W1 @ x + b1
        a2 = np.maximum(0.0, z1)
        pred = W2 @ a2 + b2
        loss = np.mean((pred - y_true)**2)

        d_pred = (2 / len(y_true)) * (pred - y_true)
        db2 = d_pred
        dW2 = np.outer(d_pred, a2)
        d_a1 = np.dot(d_pred, W2)                 # backprop through W2
        d_z1 = d_a1 * (z1 > 0)              # ReLU gate: zero out where z1 <= 0
        db1 = d_z1
        dW1 = np.outer(d_z1, x)

        return {'loss': round(float(loss), 4),
        'dW1': self.round4(dW1), 
        'db1': self.round4(db1),
        'dW2': self.round4(dW2),
        'db2': self.round4(db2)}

    def round4(self, arr):
        return (np.round(arr, 4) + 0.0).tolist()
