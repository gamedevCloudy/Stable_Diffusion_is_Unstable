# classifyer.py (updated for Hugging Face CLIP)

import os
import string
import torch
import torchvision
from PIL import Image

# Hugging Face CLIP imports
from transformers import CLIPModel, CLIPProcessor

def build_zeroshot_weights_hf(clip_model, clip_processor, classnames, templates):
    """
    Build zero-shot classifier weights for Hugging Face CLIP by:
      1) Creating template prompts (e.g., "a photo of a {}") for each class
      2) Encoding them with clip_model.get_text_features
      3) Averaging embeddings to produce a single vector per class
    Returns a tensor of shape [embed_dim, num_classes].
    """
    device = next(clip_model.parameters()).device
    clip_model.eval()

    zeroshot_weights = []
    for classname in classnames:
        # e.g., "a photo of a dog", "this is a picture of a dog", etc.
        texts = [template.format(classname) for template in templates]
        # Tokenize and get text features
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Average embeddings for this class
        class_embedding = text_embeds.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    # Stack all class embeddings => shape: [embed_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).detach()
    return zeroshot_weights.to(device)


def img_process(images, img_size):
    """
    Example function for post-processing images prior to CLIP.
    Currently uses an ROIAlign operation to extract a 224x224 region.
    Adjust if needed, or remove entirely if you rely on CLIPProcessor alone.
    """
    roiAlign = torchvision.ops.RoIAlign(
        output_size=224,
        sampling_ratio=-1,
        spatial_scale=1,
        aligned=True
    )
    batch_image = []
    coord = torch.tensor([[0.0, 0.0, float(img_size), float(img_size)]]).cuda().to(torch.float16)

    # images should be a Tensor of shape [B, C, H, W]
    # We apply roiAlign once per batch element
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)   # [1, C, H, W]
        image = roiAlign(image, [coord]).squeeze()  # [C, 224, 224]
        batch_image.append(image)

    batch_image = torch.stack(batch_image, dim=0)  # [B, C, 224, 224]
    return batch_image


def save_pil_image(image, clean_text, successful, adv_text):
    """
    Saves a PIL image (or a list of PIL images) into a folder structure:
      ./images/{clean_text}/{successful}/{adv_text}.png
    """
    file_path = os.path.join("images", clean_text, successful)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    adv_text = clean_filename(adv_text)
    save_path = os.path.join(file_path, adv_text + ".png")

    # 'image' could be a single PIL image or a list of PIL images.
    # Adjust if your pipeline outputs a single image instead.
    if isinstance(image, list):
        image[0].save(save_path)
    else:
        image.save(save_path)


def clean_filename(filename):
    """
    Remove any disallowed characters from 'filename'
    so that the result is safe to use as part of a file path.
    """
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned_filename = "".join(c for c in filename if c in valid_chars)
    return cleaned_filename
