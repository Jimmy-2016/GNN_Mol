import torch
import torch.nn as nn

# Assuming your vector has size 100
input_size = 100

# Desired output size
output_size = (20, 50)

# Calculate the required input channels for the transpose convolution
# Here, we assume the input is a single channel (1D vector), so input channels = 1
input_channels = 1

# Define the transpose convolution layer
transpose_conv1 = nn.ConvTranspose2d(input_channels, 1, kernel_size=5, stride=1, padding=0)
transpose_conv2 = nn.ConvTranspose2d(input_channels, 1, kernel_size=5, stride=1, padding=0)

# Reshape your vector (tensor) to have shape (batch_size, input_channels, 1, input_size)
# For this example, let's assume batch_size is 1
vector = torch.randn(1, input_channels, input_size)

# Apply the transpose convolution
result1 = transpose_conv1(vector)
result2 = transpose_conv1(result1)


# The result will have shape (batch_size, 1, 20, 50)
# If batch_size is 1, you can remove the first dimension
result1 = result1.squeeze(0)
result2 = result2.squeeze(0)


# Print the result
print(result1.shape)
print(result2.shape)