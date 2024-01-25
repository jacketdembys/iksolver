import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AngleVectorMSELoss(nn.Module):
    def __init__(self):
        super(AngleVectorMSELoss, self).__init__()

    def forward(self, angles1, angles2):
        # Normalize angles to the range [0, 2*pi)
        angles1 = angles1 % (2 * np.pi)
        angles2 = angles2 % (2 * np.pi)

        # Calculate the angular differences element-wise
        diff = torch.abs(angles1 - angles2)

        # Adjust for wrapping around the circle
        diff = torch.min(diff, 2 * np.pi - diff)

        # Calculate mean squared error (MSE) for each angle vector
        mse = torch.mean(torch.square(diff), dim=1)
        overall_mse = torch.mean(mse)
        
        return overall_mse

# Example batch usage with angle vectors
"""
angles1_rad = torch.tensor(np.radians([[30, 45, 60], 
                                       [120, 150, 180]]))  # Convert degrees to radians for a batch

angles2_rad = torch.tensor(np.radians([[350, 60, 180], 
                                       [100, 140, 160]]))
"""



angles1_rad = torch.tensor(np.radians([[45]]))  # Convert degrees to radians for a batch
angles2_rad = torch.tensor(np.radians([[-45]]))

# Create an instance of the custom loss function
angle_vector_loss = AngleVectorMSELoss()

# Calculate the loss (MSE for angle vector differences in batches)
loss = angle_vector_loss(angles1_rad, angles2_rad)
print("Mean Squared Error (MSE) for angle vector differences (batch):", loss.item())
