#!/usr/bin/env python
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified to use Hugging Face CLIP instead of OpenAI CLIP.

import argparse
import csv
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import jiwer
from bert_score.utils import get_idf_dict
from datasets import load_dataset

# -------------------------
# Hugging Face Transformers
# -------------------------
import transformers
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    CLIPModel,
    CLIPProcessor
)
transformers.logging.set_verbosity_error()

# -------------------------
# diffusers for Stable Diffusion
# -------------------------
from diffusers import StableDiffusionPipeline

# -------------------------
# Local modules / utils
# -------------------------
from data import cfg
from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args
from classfier import img_process, clean_filename, save_pil_image  # adapt as needed

# Turn off possible warnings
warnings.filterwarnings("ignore", category=UserWarning)


def wer(x, y):
    """
    Word Error Rate (token-level).
    """
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])
    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    """
    A simplified BERTScore approach, given reference & candidate embeddings.
    """
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)

    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens for both
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    """
    Negative log-perplexity given model logits and Gumbel-Softmax coefficients.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, : shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def check_classname(adv_text, classname):
    """
    Ensure the final adversarial text retains the target classname.
    """
    return (classname in adv_text)


def load_gpt2_model(model_name="gpt2", output_hidden_states=False):
    """
    Load a GPT-2 model from Hugging Face.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = output_hidden_states
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    return model


def build_zeroshot_weights_hf(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    classnames,
    templates
):
    """
    Build zero-shot classifier weights for Hugging Face CLIP.

    1) For each classname, fill in each template (e.g. "a photo of a {}").
    2) Encode text with CLIP.
    3) Average embeddings to get a single vector for each class.
    4) Collect these in a matrix of shape [embed_dim, num_classes].
    """
    device = next(clip_model.parameters()).device
    clip_model.eval()

    zeroshot_weights = []
    for classname in classnames:
        texts = [t.format(classname) for t in templates]
        # Process text
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Average all template embeddings for this class
        mean_embeds = text_embeds.mean(dim=0)
        mean_embeds = mean_embeds / mean_embeds.norm()
        zeroshot_weights.append(mean_embeds)

    # final shape: (embedding_dim, num_classes)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def main():
    # ---------------------------------
    # 1) Basic setup and parse arguments
    # ---------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    output_file = get_output_file(cfg["start_index"], cfg["start_index"] + cfg["num_samples"])
    output_file = os.path.join(cfg["adv_samples_folder"], output_file)
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print("Skipping batch as it has already been completed.")
        exit()

    # ---------------------------------
    # 2) Load dataset & config
    # ---------------------------------
    imagenet_classes = cfg["imagenet_classes"]       # list of class names
    imagenet_templates = cfg["imagenet_templates"]   # e.g. ["a photo of a {}", ...]
    inference_steps = cfg["num_inference_steps"]
    img_size = cfg["img_size"]
    save = cfg["save"]

    text_path = "./imageNet_short_prompt.csv"  # or "../imageNet_long_prompt.csv"
    dataset = load_data(text_path)

    # ---------------------------------
    # 3) Load models: CLIP (HF), SD, GPT-2
    # ---------------------------------
    print("Loading Hugging Face CLIP model...")
    clip_name = "openai/clip-vit-base-patch16"  # Or any HF CLIP model
    clip_model = CLIPModel.from_pretrained(clip_name).cuda()
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    # Build zero-shot weights
    print("Building zero-shot classifier weights...")
    zeroshot_weights = build_zeroshot_weights_hf(
        clip_model, clip_processor, imagenet_classes, imagenet_templates
    )  # shape: [embed_dim, #classes]

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    print("Loading GPT-2 language model...")
    ref_model = load_gpt2_model("gpt2-medium", output_hidden_states=True).cuda()
    ref_model.eval()

    # Precompute text encoder embeddings for entire vocab
    with torch.no_grad():
        embeddings = pipe.text_encoder.get_input_embeddings()(
            torch.arange(0, pipe.tokenizer.vocab_size).long().cuda()
        )
        # GPT-2 reference embeddings (for perplexity/embedding similarity)
        ref_embeddings = ref_model.get_input_embeddings()(
            torch.arange(0, ref_model.config.vocab_size).long().cuda()
        ).to(torch.float16)

    # ---------------------------------
    # 4) Tokenize dataset prompts
    # ---------------------------------
    text_key = "prompt"
    testset_key = "train"

    def preprocess_function(examples):
        return pipe.tokenizer(examples[text_key], padding="max_length", truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    if cfg["constraint"] == "bertscore_idf":
        idf_dict = get_idf_dict(dataset["train"][text_key], pipe.tokenizer, nthreads=20)
    else:
        idf_dict = None

    # ---------------------------------
    # 5) Prepare logs
    # ---------------------------------
    adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits = []
    adv_logits = []
    token_errors = []
    times = []

    end_index = min(cfg["start_index"] + cfg["num_samples"], len(encoded_dataset[testset_key]))
    adv_losses = torch.zeros(end_index - cfg["start_index"], cfg["num_iters"])
    ref_losses = torch.zeros(end_index - cfg["start_index"], cfg["num_iters"])
    perp_losses = torch.zeros(end_index - cfg["start_index"], cfg["num_iters"])
    entropies = torch.zeros(end_index - cfg["start_index"], cfg["num_iters"])

    torch.autograd.set_detect_anomaly(True)

    # ---------------------------------
    # 6) Main loop over prompts
    # ---------------------------------
    for idx in range(cfg["start_index"], end_index):
        clean_text = encoded_dataset[testset_key]["prompt"][idx]
        input_ids = encoded_dataset[testset_key]["input_ids"][idx]
        input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).cuda()

        # Encode text for SD
        prompt_embeddings = pipe.text_encoder(input_ids=input_ids_tensor)[0]
        prompt_embeddings = prompt_embeddings.to(dtype=pipe.text_encoder.dtype)

        # The "ImageNet class label" for this prompt
        label = encoded_dataset[testset_key]["label"][idx]
        classname = encoded_dataset[testset_key]["classname"][idx]

        # -- Evaluate original prompt in stable diffusion & CLIP
        with torch.no_grad():
            images = pipe(prompt_embeds=prompt_embeddings, num_inference_steps=inference_steps).images
            # Convert to tensor for CLIP
            images_tensor = img_process(images[0], img_size)  # your custom function
            # HF CLIP forward
            clip_inputs = clip_processor(images=images[0], return_tensors="pt").to("cuda")
            image_embeds = clip_model.get_image_features(**clip_inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Zero-shot classification => dot product with zeroshot_weights
            # shape: image_embeds [1, embed_dim], zeroshot_weights [embed_dim, #classes]
            logits_per_image = 100.0 * image_embeds @ zeroshot_weights
            clean_logit = logits_per_image.squeeze(0)  # shape [#classes]

        if clean_logit.argmax().item() != label:
            print("Skipping text, clean prompt misclassified.")
            continue

        # Save clean image if desired
        if save:
            class_filename = clean_filename(clean_text)
            save_pil_image(images[0], class_filename, "clean", classname)

        print(f"\nIndex {idx} | LABEL: {label}")
        print(f"CLEAN TEXT: {pipe.tokenizer.decode(input_ids, skip_special_tokens=True)}")
        print(f"CLASSNAME: {classname}")

        # Mark tokens we want to forbid from changing
        forbidden = np.zeros(len(input_ids)).astype(bool)
        # For example, forbid all original tokens from altering:
        unchange_ids = pipe.tokenizer(clean_text, truncation=True)["input_ids"]
        for i, token_id in enumerate(input_ids):
            if token_id in unchange_ids[:-1]:
                forbidden[i] = True
        forbidden_indices = torch.from_numpy(np.arange(0, len(input_ids))[forbidden]).cuda()

        # ---------------------------------
        # 6.1 Initialize Gumbel-Softmax log coeffs
        # ---------------------------------
        log_coeffs = torch.zeros(len(input_ids), embeddings.size(0)).cuda()
        indices = torch.arange(log_coeffs.size(0)).long().cuda()
        # Original tokens get a high initial coeff
        log_coeffs[indices, torch.LongTensor(input_ids).cuda()] = cfg["initial_coeff"]
        log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=cfg["lr"])
        start_time = time.time()

        # Reference GPT-2 output for constraints
        with torch.no_grad():
            orig_output = ref_model(torch.LongTensor(input_ids).unsqueeze(0).cuda()).hidden_states[
                cfg["embed_layer"]
            ]
            if cfg["constraint"] == "bertscore_idf":
                ref_weights = torch.FloatTensor([idf_dict[x] for x in input_ids]).cuda()
                ref_weights /= ref_weights.sum()
            else:
                ref_weights = None

        # ---------------------------------
        # 6.2 Gradient Descent Loop
        # ---------------------------------
        for i in range(cfg["num_iters"]):
            optimizer.zero_grad()
            coeffs = F.gumbel_softmax(
                log_coeffs.unsqueeze(0).repeat(cfg["batch_size"], 1, 1), hard=False
            )  # shape: [B, T, V]

            with torch.autocast("cuda", torch.float16):
                # Convert Gumbel distribution -> text encoder input
                inputs_embeds = coeffs @ embeddings[None, :, :]  # [B, T, D]
                # Pass to SD text encoder
                # inputs_embeds = pipe.text_encoder(inputs_embeds=inputs_embeds)[0]

                # Generate images
                images = pipe(
                    prompt_embeds=inputs_embeds,
                    num_inference_steps=inference_steps,
                    height=img_size,
                    width=img_size,
                ).images

                # Prepare image for CLIP
                clip_inputs = clip_processor(images=images[0], return_tensors="pt").to("cuda")
                image_embeds = clip_model.get_image_features(**clip_inputs)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                # Dot product with zero-shot weights
                pred = 100.0 * image_embeds @ zeroshot_weights  # shape [1, #classes]

                # Adversarial loss (CE or CW)
                if cfg["adv_loss"] == "ce":
                    adv_loss = -F.cross_entropy(pred, torch.tensor([label]).long().cuda())
                elif cfg["adv_loss"] == "cw":
                    top_preds = pred.sort(descending=True)[1]
                    correct = (top_preds[:, 0] == label).long()
                    indices_ = top_preds.gather(1, correct.view(-1, 1))
                    adv_loss = (pred[:, label] - pred.gather(1, indices_).squeeze() + cfg["kappa"]).clamp(min=0).mean()
                else:
                    adv_loss = torch.tensor(0.0).cuda()

                # Similarity constraint (GPT-2 embeddings)
                ref_input_embeds = coeffs @ ref_embeddings[None, :, :]
                pred_ref = ref_model(inputs_embeds=ref_input_embeds)
                if cfg["lam_sim"] > 0:
                    output = pred_ref.hidden_states[cfg["embed_layer"]]
                    if cfg["constraint"].startswith("bertscore"):
                        ref_loss = -cfg["lam_sim"] * bert_score(orig_output, output, weights=ref_weights).mean()
                    else:
                        # e.g., simple cosine similarity
                        output = output[:, -1]
                        cosine = (output * orig_output).sum(1) / (output.norm(2, 1) * orig_output.norm(2, 1))
                        ref_loss = cfg["lam_sim"] * cosine.mean()
                else:
                    ref_loss = torch.tensor(0.0).cuda()

                # Perplexity constraint
                if cfg["lam_perp"] > 0:
                    perp_loss = cfg["lam_perp"] * log_perplexity(pred_ref.logits, coeffs)
                else:
                    perp_loss = torch.tensor(0.0).cuda()

            total_loss = adv_loss + ref_loss + perp_loss
            total_loss.backward()

            # Zero out forbidden
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()

            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
            if i % cfg["print_every"] == 0:
                print(
                    f"Iter {i+1}/{cfg['num_iters']} | total={total_loss.item():.4f}, "
                    f"adv={adv_loss.item():.4f}, ref={ref_loss.item():.4f}, "
                    f"perp={perp_loss.item():.4f}, ent={entropy.item():.4f}"
                )

            adv_losses[idx - cfg["start_index"], i] = adv_loss.detach().item()
            ref_losses[idx - cfg["start_index"], i] = ref_loss.detach().item()
            perp_losses[idx - cfg["start_index"], i] = perp_loss.detach().item()
            entropies[idx - cfg["start_index"], i] = entropy.detach().item()

        times.append(time.time() - start_time)

        # ---------------------------------
        # 6.3 Final adversarial sampling
        # ---------------------------------
        print("\nCLEAN TEXT:", pipe.tokenizer.decode(input_ids, skip_special_tokens=True))
        clean_texts.append(clean_text)
        clean_logits.append(clean_logit.detach())

        print("ADVERSARIAL TEXT SAMPLES:")
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            for j in range(cfg["gumbel_samples"]):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
                adv_ids = adv_ids.cpu().tolist()
                adv_text = pipe.tokenizer.decode(adv_ids, skip_special_tokens=True)

                print(f"  Sample {j}: {adv_text}")
                x = pipe.tokenizer(adv_text, truncation=True, return_tensors="pt")
                token_errors.append(wer(adv_ids, x["input_ids"][0]))

                # Generate image
                images = pipe(
                    adv_text,
                    num_inference_steps=inference_steps,
                    height=img_size,
                    width=img_size,
                ).images
                clip_inputs = clip_processor(images=images[0], return_tensors="pt").to("cuda")
                image_embeds = clip_model.get_image_features(**clip_inputs)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                adv_logit = 100.0 * image_embeds @ zeroshot_weights

                # Check success
                if adv_logit.argmax().item() != label and check_classname(adv_text, classname):
                    successful = "success"
                    adv_texts.append(adv_text)
                    adv_logits.append(adv_logit)
                    if save:
                        save_pil_image(images[0], class_filename, successful, str(j))
                else:
                    successful = "failed"
                    if save:
                        save_pil_image(images[0], class_filename, successful, str(j))

                # Optional CSV logging
                if save:
                    csv_path = "adv_text.csv"
                    mode = "a" if os.path.exists(csv_path) else "w"
                    with open(csv_path, mode, newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        if mode == "w":
                            csv_writer.writerow(["clean_text", "successful", "sample_j", "adv_text"])
                        csv_writer.writerow([class_filename, successful, j, adv_text])

        adv_log_coeffs.append(log_coeffs.cpu())

    # ---------------------------------
    # 7) Final save
    # ---------------------------------
    avg_token_err = sum(token_errors) / len(token_errors) if len(token_errors) else 0.0
    print(f"\nToken Error Rate: {avg_token_err:.4f} over {len(token_errors)} tokens")

    # Flatten logs if not empty
    final_adv_logits = torch.cat(adv_logits, 0) if len(adv_logits) > 0 else torch.empty(0)
    final_clean_logits = torch.cat(clean_logits, 0) if len(clean_logits) > 0 else torch.empty(0)

    torch.save(
        {
            "adv_log_coeffs": adv_log_coeffs,
            "adv_logits": final_adv_logits,
            "adv_losses": adv_losses,
            "adv_texts": adv_texts,
            "clean_logits": final_clean_logits,
            "clean_texts": clean_texts,
            "entropies": entropies,
            "labels": list(encoded_dataset[testset_key]["label"][cfg["start_index"] : end_index]),
            "perp_losses": perp_losses,
            "ref_losses": ref_losses,
            "times": times,
            "token_error": token_errors,
        },
        output_file,
    )

    print("\nAdversarial attack process complete!")


if __name__ == "__main__":
    main()
