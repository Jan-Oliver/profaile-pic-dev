import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import hashlib
import itertools
import contextlib
import random
import json
import io
import PIL
import math
import sys
import os
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import boto3
import sagemaker
from sagemaker import get_execution_role
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusionDepth2ImgPipeline, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, DPTFeatureExtractor, DPTForDepthEstimation
from diffusers.utils.pil_utils import PIL_INTERPOLATION
torch.backends.cudnn.benchmark = True


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")

    # AWS Sagemaker Params
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # For this script we don't need the default hyperparameters provided in the sagemaker examples.
    # Output directory is where we will store the model into. This is the path that sagemaker expects the model to be.
    # If we specify a different value in estimator.fit we will overwrite the environment variable.
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="",
        help="The output directory where the final model will be written to.",
    )
    parser.add_argument(
        "--pretrained_depth_model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        required=False,
        help="Path to pretrained depth model in AWS S3 bucket: stabilityai/stable-diffusion-2-1-base. This model will be finetuned",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Path to pretrained vae in AWS S3 bucket: stabilityai/sd-vae-ft-mse",
    )
    # Checkpointing configuration
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="/opt/ml/checkpoints",
        help="The output directory where the model checkpoints will be written to.",
    )
    parser.add_argument(
        "--checkpointing_interval",
        type=int,
        default=250,
        help="Safe model every N steps of training",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="fp16",
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # Defines the directories where the class and instance images are stored.
    # Also defines the text input (prompt) that is fed into the model together with the class / instance images
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="",
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="",
        help="A folder containing the training data of class images.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )

    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    # Parameters for Prior Preservation
    # Allows to keep the prior of the model
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        type=bool,
        help="Flag to add prior preservation loss.",
    )

    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."),
    )
    parser.add_argument(
        "--class_batch_size",
        type=int, default=4, help="Batch size (per device) for sampling images."
    )

    # Training specific hyperparameters
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_text_encoder",
        default=True,
        type=bool,
        help="Whether to train the text encoder or not. If argument is added the text encoder will also be trained."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3500,
        help="Total number of training steps to perform",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )

    # Hyperparameters for optimization regarding GPU usage
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        type=bool,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        default=True,
        type=bool,
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--not_cache_latents",
        default=False,
        type=bool,
        help="Do not precompute and cache latents from VAE."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        concept,
        with_prior_preservation=True,
        num_class_images=None,
    ):
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation

        self.instance_images_path = []
        self.class_images_path = []

        inst_img_path = [(x, concept["instance_prompt"]) for x in Path(
            concept["instance_data_dir"]).iterdir() if x.is_file()]
        self.instance_images_path.extend(inst_img_path)

        if with_prior_preservation:
            class_img_path = [(x, concept["class_prompt"]) for x in Path(
                concept["class_data_dir"]).iterdir() if x.is_file()]
            self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_images_path[index %
                                                                   self.num_instance_images]
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.preprocess(instance_image)[0]
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.with_prior_preservation:
            class_path, class_prompt = self.class_images_path[index %
                                                              self.num_class_images]
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.preprocess(class_image)[0]
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_random_hash():
    return uuid.uuid4().hex


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    environment_variables = os.environ
    logger.info("Environment variables", environment_variables)

    # Set seed for reproducible training
    set_seed(args.seed)

    # Define concept that will be learned
    concept = {
        "instance_prompt": args.instance_prompt,
        "class_prompt": args.class_prompt,
        "instance_data_dir": args.instance_data_dir,
        "class_data_dir": args.class_data_dir
    }

    class_images_dir = Path(concept["class_data_dir"])

    # Get Width and Height of images
    filename = os.listdir(class_images_dir)[0]
    image = Image.open(os.path.join(class_images_dir, filename))
    WIDTH, HEIGHT = image.size

    # Check if there are folders in /models
    if len(os.listdir(args.model_checkpoint_dir)) != 0:
        folders = sorted([f for f in os.listdir(
            args.model_checkpoint_dir) if f != "0"], key=lambda x: int(x))
        checkpoint_epoch = int(folders[-1])
        logger.info(
            f"***** continue training from checkpoint {checkpoint_epoch} *****")
        args.pretrained_depth_model_path = os.path.join(
            args.model_checkpoint_dir, str(checkpoint_epoch))
    else:
        logger.info(f"***** training from original pretrained model *****")
        checkpoint_epoch = 0

    # Load the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_depth_model_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_depth_model_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_depth_model_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_depth_model_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters(
        )) if args.train_text_encoder else unet.parameters()
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_depth_model_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        concept=concept,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        num_class_images=args.num_class_images,
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=8
    )

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(
                    accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(
                    batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(
                        text_encoder(batch["input_ids"])[0])

        train_dataset = LatentsDataset(
            latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    def save_checkpoint(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(
                    args.pretrained_depth_model_path, subfolder="text_encoder", revision=args.revision)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                      beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

            pipeline = StableDiffusionPipeline.from_pretrained(
                # MAYBE: The author also moved unwarp_model(unet).to(torch.float16) and text_enc_model.to(torch.float16)
                args.pretrained_depth_model_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_path,
                    subfolder=None,
                    revision=None,
                ),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision=args.revision,
            )
            save_dir = os.path.join(args.model_checkpoint_dir, f"{step}")
            pipeline.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"[*] Weights saved at {save_dir}")

    def save_final_model():
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(
                    args.pretrained_depth_model_path, subfolder="text_encoder", revision=args.revision)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                      beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_depth_model_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_path,
                    subfolder=None,
                    revision=None,
                ),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision=args.revision,
            )
            pipeline.save_pretrained(args.model_output_dir)
            with open(os.path.join(args.model_output_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

    # Train!
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    progress_bar.update(checkpoint_epoch)
    global_step = checkpoint_epoch
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(num_train_epochs):
        epoch = epoch + checkpoint_epoch
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        if args.not_cache_latents:
            random.shuffle(train_dataset.class_images_path)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(
                            dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(batch[0][1])[
                                0]
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = text_encoder(
                            batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(
                        noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(
                    ), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(),
                                      noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not global_step % args.log_interval:
                logs = {"loss": loss_avg.avg.item(
                ), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step > 0 and not global_step % args.checkpointing_interval and not global_step == args.max_train_steps:
                save_checkpoint(global_step)

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    save_final_model()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
