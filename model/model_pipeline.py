import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16_bn

import wandb
import matplotlib.pyplot as plt

from utils.early_stopping import EarlyStopping
from utils.helper_functions import convert_pixels_to_mm, get_length_line_simple, dice_loss, CombinedLoss, ModelFC

torch.hub.set_dir('/cs/student/projects1/2021/izaffar/.cache/torch/hub')

DATA_DIR = "../data"
# for filename in os.listdir(DATA_DIR):
#     print(filename)

def set_seed(seed):
    # Set random seed
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" # Cublas ting

def make(config, device):
    # Make the model
    model = get_model(config["model_output"], config["model_arch"], device)

    # Make the loss and optimizer
    loss_fns = {
        "MSE": nn.MSELoss(),
        "BCE": nn.BCELoss(),
        "Huber": nn.HuberLoss(),
        "Dice": dice_loss
    }
    if config["criterion"] == "Combined":
        criterion = CombinedLoss(alpha=config["criterion_alpha"],
                                 mask_loss_fn=loss_fns[config["criterion_mask"]],
                                 length_loss_fn=loss_fns[config["criterion_length"]],
                                 threshold_epochs=config["criterion_threshold_epochs"],
                                 threshold_length_loss=config["criterion_threshold_length_loss"])
    else:
        criterion = loss_fns[config["criterion"]]
    print("Criterion:", criterion)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"])
    
    return model, criterion, optimizer

# Load the pre-trained model
def get_model(model_output, model_arch, device="cpu"):
    if model_arch == "brain_mri":
        pretrained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=32, pretrained=True)
        new_in_channels = 1
        modified_encoder1_weight = pretrained_model.encoder1.enc1conv1.weight.data[:, :new_in_channels, :, :]
        pretrained_model.encoder1.enc1conv1 = nn.Conv2d(new_in_channels, 32, kernel_size=3, padding=1)
        pretrained_model.encoder1.enc1conv1.weight.data = modified_encoder1_weight

    if model_output == "mask":
        return pretrained_model.to(device)
    elif model_output == "length":
        return ModelFC(base_model=pretrained_model).to(device)

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=0)
    return loader

def train(model, train_loader, val_loader, criterion, optimizer, run, config, device):
    train_losses, val_losses = [], []
    val_mean_diffs, val_std_diffs = [], []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, delta=0.0001, path=f"checkpoint.pt", verbose=False)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for _, (images, masks) in enumerate(train_loader):
            loss = train_batch(images, masks, model, optimizer, criterion, epoch, config, device)
            train_loss += loss.item() * len(images)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_diffs = []

        with torch.no_grad():
            for _, (images, masks) in enumerate(val_loader):
                loss, diffs = val_batch(images, masks, model, criterion, epoch, config, device)
                val_loss += loss.item() * len(images)
                val_diffs.extend(diffs.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_mean_diffs.append(np.mean(val_diffs))
        val_std_diffs.append(np.std(val_diffs))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Log metrics
        run.log({"epoch": epoch,
                    f"train_loss": train_loss,
                    f"val_loss": val_loss,
                    f"val_mean_diff": val_mean_diffs[-1],
                    f"val_std_diff": val_std_diffs[-1]},
                    step=epoch)
        print(f"Epoch {str(epoch).zfill(3)} - Train: {train_loss:.3f}, Val: {val_loss:.3f}")
        print(f"Val Mean Diff: {val_mean_diffs[-1]:.3f}, Val Std Diff: {val_std_diffs[-1]:.3f}")

    return train_losses, val_losses, val_mean_diffs, val_std_diffs

def train_batch(images, masks, model, optimizer, criterion, epoch, config, device):
    images, masks = images.to(device), masks.to(device)
    target_lengths = get_length_line_simple(masks)
    target_lengths_mm = convert_pixels_to_mm(target_lengths, masks).to(device)

    # Forward pass ➡
    outputs = model(images)

    if config["model_output"] == "length":
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, target_lengths_mm)
    elif config["model_output"] == "mask":
        pred_masks = (outputs > config["mask_threshold"]).float()
        pred_lengths = get_length_line_simple(pred_masks)
        pred_lengths_mm = convert_pixels_to_mm(pred_lengths, pred_masks).to(device)
        
        criterion.trigger_combined(epoch, pred_lengths_mm, target_lengths_mm)
        loss = criterion(outputs.requires_grad_(), pred_lengths_mm.requires_grad_(), masks.requires_grad_(), target_lengths_mm.requires_grad_(), epoch)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def val_batch(images, masks, model, criterion, epoch, config, device):
    images, masks = images.to(device), masks.to(device)
    target_lengths = get_length_line_simple(masks)
    target_lengths_mm = convert_pixels_to_mm(target_lengths, masks).to(device)
    
    # Forward pass ➡
    outputs = model(images)

    if config["model_output"] == "length":
        outputs = outputs.squeeze(1)
        val_loss = criterion(outputs, target_lengths_mm)
        val_diffs = torch.abs(outputs - target_lengths_mm)
    elif config["model_output"] == "mask":
        pred_masks = (outputs > config["mask_threshold"]).float()
        pred_lengths = get_length_line_simple(pred_masks)
        pred_lengths_mm = convert_pixels_to_mm(pred_lengths, pred_masks).to(device)
        val_loss = criterion(outputs, pred_lengths_mm, masks, target_lengths_mm, epoch)
        val_diffs = torch.abs(pred_lengths_mm - target_lengths_mm)
        # TODO: Log Dice score as well

    return val_loss, val_diffs

def test(model, test_loader, run, config, device):
    # Load best model
    model.load_state_dict(torch.load("checkpoint.pt"))

    images_list, masks_list, lengths_list, pred_lengths_list, pred_masks_list, diffs_list = evaluate_model(model, test_loader, config, device)

    mean_diff = np.mean(diffs_list)
    std_diff = np.std(diffs_list)
    # print(f"Metrics for {len(images_list)} test images - Mean Diff: {mean_diff}mm, Std Diff: {std_diff}mm")

    run.log({"mean_diff": mean_diff, "std_diff": std_diff})

    visualize_results(images_list, masks_list, lengths_list, pred_lengths_list, pred_masks_list, diffs_list, run, config, save=True)

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, images, "model.onnx")
    # wandb.save("model.onnx")

    return mean_diff, std_diff


def evaluate_model(model, dataloader, config, device):
    images_list = []
    masks_list = []
    # outputs_list = []
    lengths_list = []
    pred_lengths_list = []
    pred_masks_list = []
    diffs_list = []
    # gradcam_list = []

    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            target_lengths = get_length_line_simple(masks)
            target_lengths_mm = convert_pixels_to_mm(target_lengths, masks).to(device)

            outputs = model(images)
            if config["model_output"] == "length":
                outputs = outputs.squeeze(1)
                diffs = torch.abs(outputs - target_lengths_mm)

                pred_lengths_list.extend(outputs.cpu().detach().numpy())
            elif config["model_output"] == "mask":
                pred_masks = (outputs > config["mask_threshold"]).float()
                pred_lengths = get_length_line_simple(pred_masks)
                pred_lengths_mm = convert_pixels_to_mm(pred_lengths, pred_masks).to(device)
                pred_lengths_list.extend(pred_lengths_mm.cpu().detach().numpy())
                diffs = torch.abs(pred_lengths_mm - target_lengths_mm)

                # pred_masks_list.extend(outputs.squeeze(1).cpu().detach().numpy())
                pred_masks_list.extend(pred_masks.squeeze(1).cpu().detach().numpy())
            
            # Convert tensors to lists or numpy arrays
            images_list.extend(images.squeeze(1).cpu().detach().numpy())
            masks_list.extend(masks.squeeze(1).cpu().detach().numpy())
            lengths_list.extend(target_lengths_mm.cpu().detach().numpy())
            diffs_list.extend(diffs.cpu().detach().numpy())

    return images_list, masks_list, lengths_list, pred_lengths_list, pred_masks_list, diffs_list

def visualize_results(images_list, masks_list, lengths_list, pred_lengths_list, pred_masks_list, diffs_list, run, config, save=False):
    num_images = len(images_list)
    images_array = []

    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        axs[0].imshow(images_list[i], cmap="gray")
        axs[0].set_title(f"Image {i}, Diff: {diffs_list[i]:.2f}")

        if config["model_output"] == "length":
            axs[1].imshow(images_list[i], cmap="gray")
            axs[1].imshow(masks_list[i], cmap="gray", interpolation="none", alpha=0.7)
            axs[1].set_title(f"Ground Truth: {lengths_list[i]:.2f}")

            # axs[2].imshow(images_list[i], cmap="gray")
            # axs[2].imshow(masks_list[i], cmap="gray", interpolation="none", alpha=0.7)
            axs[2].set_title(f"Output: {pred_lengths_list[i]:.2f}")
        elif config["model_output"] == "mask":
            axs[1].imshow(images_list[i], cmap="gray")
            axs[1].imshow(masks_list[i], cmap="gray", interpolation="none", alpha=0.7)
            axs[1].set_title(f"Ground Truth: {lengths_list[i]:.2f}")

            axs[2].imshow(images_list[i], cmap="gray")
            axs[2].imshow(pred_masks_list[i], cmap="gray", interpolation="none", alpha=0.7)
            axs[2].set_title(f"Output: {pred_lengths_list[i]:.2f}")

        for ax in axs:
            ax.axis("off")
        fig.tight_layout()

        if save:
            fig_path = f"results/output_{i}.png"
            fig.savefig(fig_path)
            images_array.append(wandb.Image(fig_path))
            plt.close(fig)
        else:
            images_array.append(fig)

    if save:
        run.log({"preds": images_array})
    else:
        plt.show()
