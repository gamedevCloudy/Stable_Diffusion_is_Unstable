# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
from tqdm import tqdm
import torch

# Modern Hugging Face imports
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AutoConfig,
    GPT2Tokenizer,
)

from data import cfg

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

# Offset target by 1 if labels start from 1
def target_offset(examples):
    examples["label"] = [x - 1 for x in examples["label"]]
    return examples

def get_output_file(start, end):
    """
    Returns the output file name based on config options.
    """
    kappa = cfg['kappa']
    num_iters = cfg['num_iters']
    lam_sim = cfg['lam_sim']
    lam_perp = cfg['lam_perp']
    embed_layer = cfg['embed_layer']
    constraint = cfg['constraint']

    attack_str = cfg['adv_loss']
    if cfg['adv_loss'] == 'cw':
        attack_str += f'_kappa={kappa}'

    output_file = f"{start}-{end}"
    output_file += f"_iters={num_iters}_{attack_str}_lambda_sim={lam_sim}_lambda_perp={lam_perp}_emblayer={embed_layer}_{constraint}.pth"
    return output_file


def load_checkpoints(args):
    """
    Example function that loads adversarial checkpoints from disk.
    """
    if args.dataset == 'mnli':
        adv_log_coeffs = {'premise': [], 'hypothesis': []}
        clean_texts = {'premise': [], 'hypothesis': []}
        adv_texts = {'premise': [], 'hypothesis': []}
    else:
        adv_log_coeffs, clean_texts, adv_texts = [], [], []

    clean_logits, adv_logits, times, labels = [], [], [], []

    for i in tqdm(range(args.start_index, args.end_index, args.num_samples)):
        output_file = get_output_file(i, i + args.num_samples)
        output_file = os.path.join(args.adv_samples_folder, output_file)
        if os.path.exists(output_file):
            checkpoint = torch.load(output_file)
            clean_logits.append(checkpoint['clean_logits'])
            adv_logits.append(checkpoint['adv_logits'])
            labels += checkpoint['labels']
            times += checkpoint['times']

            if args.dataset == 'mnli':
                adv_log_coeffs['premise'] += checkpoint['adv_log_coeffs']['premise']
                adv_log_coeffs['hypothesis'] += checkpoint['adv_log_coeffs']['hypothesis']
                clean_texts['premise'] += checkpoint['clean_texts']['premise']
                clean_texts['hypothesis'] += checkpoint['clean_texts']['hypothesis']
                adv_texts['premise'] += checkpoint['adv_texts']['premise']
                adv_texts['hypothesis'] += checkpoint['adv_texts']['hypothesis']
            else:
                adv_log_coeffs += checkpoint['adv_log_coeffs']
                clean_texts += checkpoint['clean_texts']
                adv_texts += checkpoint['adv_texts']
        else:
            print('Skipping %s' % output_file)

    clean_logits = torch.cat(clean_logits, 0)
    adv_logits = torch.cat(adv_logits, 0)
    return clean_texts, adv_texts, clean_logits, adv_logits, adv_log_coeffs, labels, times


def print_args(args):
    """
    Utility function that pretty-prints argument values.
    """
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")


def embedding_from_weights(w):
    """
    Helper function that creates an embedding layer from given weights.
    Used if you have a custom embedding matrix.
    """
    layer = torch.nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w
    return layer


def load_gpt2_model(model_name="gpt2", output_hidden_states=False):
    """
    Modern approach to load a GPT-2 model from Hugging Face,
    rather than from a custom local checkpoint dict.
    """
    # If you need a special config:
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = output_hidden_states

    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    return model


def main():
    """
    Example usage of the new GPT-2 loading method.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--adv_samples_folder", type=str, default="adv_samples")
    args = parser.parse_args()

    # Print arguments
    print_args(args)

    # 1. Load your GPT-2 model (no custom local dict!)
    gpt2_model = load_gpt2_model(model_name="gpt2-medium", output_hidden_states=True)
    gpt2_model.eval()
    gpt2_model.cuda()

    # 2. (Optional) Load the GPT2 tokenizer if needed
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

    # 3. Example usage
    example_input = "Hello, how are you?"
    inputs = tokenizer(example_input, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        logits = outputs.logits
        hidden_states = outputs.hidden_states  # if output_hidden_states=True

    # 4. Load checkpoints or other data, as in your original script
    # (Only if you actually need them)
    # clean_texts, adv_texts, clean_logits, adv_logits, adv_log_coeffs, labels, times = load_checkpoints(args)
    # ... do something with them ...

    print("GPT-2 inference complete!")
    print("Example input:", example_input)
    print("Logits shape:", logits.shape)
    if hidden_states is not None:
        print(f"Number of hidden states: {len(hidden_states)}")


if __name__ == "__main__":
    main()
