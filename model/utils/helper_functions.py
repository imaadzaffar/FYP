import torch
import torch.nn as nn

def convert_pixels_to_mm(length_pixels, images, required_image_width_pixels=320, scale_factor=0.56525):
    image_width_pixels = images.size(2)
    length_mm = length_pixels * (required_image_width_pixels / image_width_pixels) * scale_factor
    return length_mm

def get_length_line_simple(masks):
    batch_size, _, image_width, image_height = masks.size()
    masks = masks.squeeze(1)
    lengths = []

    for i in range(batch_size):
        mask = masks[i]  # Select the mask for the current sample
        
        # Find indices of non-zero elements (where the mask is 1)
        nonzero_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(nonzero_indices) > 0:
            # Compute the bounding box from the non-zero indices
            min_x = nonzero_indices[:, 1].min().item()
            min_y = nonzero_indices[:, 0].min().item()
            max_x = nonzero_indices[:, 1].max().item()
            max_y = nonzero_indices[:, 0].max().item()

            # Calculate the diagonal length of the bounding box
            width = max_x - min_x
            height = max_y - min_y
            diagonal_length = torch.sqrt(torch.tensor(width**2 + height**2, dtype=torch.float32))
            lengths.append(diagonal_length)
        else:
            lengths.append(torch.tensor(0.0))  # If no non-zero elements found, return length 0

    return torch.stack(lengths)

# Define your new model by extending the loaded model
class ModelFC(nn.Module):
    def __init__(self, base_model):
        super(ModelFC, self).__init__()
        self.base_model = base_model

        # TODO: make sure output features is dynamic
        output_features = 256 * 256
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_features, 1)  # Fully connected layer with a single node
        )

    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model(x)

        # Apply your new layer
        x = self.final(x)

        return x
