import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        if activation == 'sigmoid':
            A1 = self.sigmoid(np.dot(x, w) + b)
        else:
            A1 = self.relu(np.dot(x, w) + b)
        
        return float(np.round(A1, 5))
    def sigmoid(self, Z1):
        return 1/(1+np.exp(-Z1))
    
    def relu(self, Z1):
        return max(0, Z1)