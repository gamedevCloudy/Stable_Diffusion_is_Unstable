# classifyer.py (final updated for Hugging Face CLIP + fixed img_process)

import os
import string
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


# If you need them directly:
from transformers import CLIPModel, CLIPProcessor


def build_zeroshot_weights_hf(clip_model, clip_processor, classnames, templates):
    """
    Build zero-shot classifier weights for Hugging Face CLIP by:
      1) Creating template prompts (e.g., "a photo of a {}") for each class
      2) Encoding them with clip_model.get_text_features
      3) Averaging embeddings to produce a single vector per class.
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
            text_embeds = clip_model.get_text_features(**inputs)  # [num_prompts, embed_dim]
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Average embeddings for this class
        class_embedding = text_embeds.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    # Stack => shape: [embed_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).detach()
    return zeroshot_weights.to(device)



def img_process(images, img_size):
    """
    Convert PIL image(s) -> Tensor [B, C, H, W] -> Move to CUDA -> ROIAlign -> [B, C, 224, 224].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Convert single PIL or list-of-PIL to a batch tensor
    if isinstance(images, Image.Image):
        # single PIL => make a list
        images = [images]

    if isinstance(images, list) and isinstance(images[0], Image.Image):
        to_tensor = T.ToTensor()  # [C,H,W]
        tensor_list = []
        for pil_img in images:
            t = to_tensor(pil_img).unsqueeze(0)  # [1,C,H,W]
            tensor_list.append(t)
        images = torch.cat(tensor_list, dim=0)  # [B,C,H,W]

    if not isinstance(images, torch.Tensor):
        raise TypeError("img_process expects a torch.Tensor or list of PIL images.")

    # 2) Now move images to CUDA if available
    images = images.to(device)

    # 3) Create ROIAlign
    roiAlign = torchvision.ops.RoIAlign(
        output_size=224,
        sampling_ratio=-1,
        spatial_scale=1,
        aligned=True,
    )

    B = images.shape[0]
    batch_image = []

    # 4) Build a bounding box on the same device as images
    coord = torch.tensor(
        [[0.0, 0.0, float(img_size), float(img_size)]],
        device=device,
        dtype=torch.float32
    )

    # 5) Loop over each image, apply ROIAlign
    for i in range(B):
        # image_i is shape [1,C,H,W]
        image_i = images[i].unsqueeze(0)
        # ROIAlign => [C,224,224]
        roi = roiAlign(image_i, [coord]).squeeze(0)
        batch_image.append(roi)

    # 6) Stack => [B,C,224,224]
    batch_image = torch.stack(batch_image, dim=0)
    return batch_image

def save_pil_image(image, clean_text, successful, adv_text):
    """
    Saves a PIL image (or list of PIL images) to:
      ./images/{clean_text}/{successful}/{adv_text}.png
    """
    file_path = os.path.join("images", clean_text, successful)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    adv_text = clean_filename(adv_text)
    save_path = os.path.join(file_path, adv_text + ".png")

    if isinstance(image, list):
        # if we get a list of PIL images
        image[0].save(save_path)
    elif isinstance(image, Image.Image):
        # single PIL image
        image.save(save_path)
    else:
        raise TypeError("save_pil_image expects a PIL Image or list of PIL Images.")


def clean_filename(filename):
    """
    Remove disallowed characters from 'filename'
    so that it is safe to use in a file path.
    """
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    return "".join(c for c in filename if c in valid_chars)
