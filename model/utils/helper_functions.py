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

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() + smooth
    dice_score = (2 * intersection) / union
    return 1 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, adaptive=True, alpha=1.0, threshold_epochs=50, threshold_length_loss=100.0, mask_loss_fn=nn.BCELoss(), length_loss_fn=nn.MSELoss()):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.adaptive = adaptive
        self.threshold_epochs = threshold_epochs
        self.threshold_length_loss = threshold_length_loss
        self.mask_loss_fn = mask_loss_fn
        self.length_loss_fn = length_loss_fn
        self.combined = False

    def trigger_combined(self, epoch, pred_lengths, target_lengths):
        length_loss = self.length_loss_fn(pred_lengths, target_lengths)
        length_loss /= self.threshold_length_loss # Scale the length loss to be in the same range as the mask loss
        if epoch > self.threshold_epochs and length_loss < 1.0:
            self.combined = True
        else:
            self.combined = False

    def forward(self, pred, pred_lengths, targets, target_lengths, epoch):
        mask_loss = self.mask_loss_fn(pred, targets)
        length_loss = self.length_loss_fn(pred_lengths, target_lengths)
        length_loss /= self.threshold_length_loss # Scale the length loss to be in the same range as the mask loss

        # print(f"Mask Loss: {mask_loss}, Length Loss: {length_loss}")
        # print(mask_loss.grad_fn, length_loss.grad_fn)

        # Combine both losses
        if self.combined:
            combined_loss = mask_loss + self.alpha * length_loss
        else:
            combined_loss = mask_loss
        return combined_loss

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
