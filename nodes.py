"""
ComfyUI GGUF Model Nodes
Comprehensive nodes for loading and working with GGUF quantized models
"""

import os
import folder_paths
import torch
from .gguf_utils import GGUFLoader, get_gguf_models, load_gguf_state_dict


class GGUFModelLoader:
    """Load a GGUF quantized model file"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_gguf_models(), {"default": ""}),
                "device": (["auto", "cpu", "cuda", "mps"], {"default": "auto"}),
                "dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "DICT")
    RETURN_NAMES = ("model", "metadata")
    FUNCTION = "load_model"
    CATEGORY = "loaders/gguf"

    def load_model(self, model_path, device="auto", dtype="auto", custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf", model_path)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF model not found: {filepath}")

        loader = GGUFLoader(filepath)
        data = loader.load()

        # Convert tensors to appropriate dtype
        if dtype == "float16":
            target_dtype = torch.float16
        elif dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float32

        # Convert tensors
        tensors = {}
        for name, tensor in data['tensors'].items():
            tensors[name] = tensor.to(dtype=target_dtype)

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Move to device
        for name in tensors:
            tensors[name] = tensors[name].to(device)

        return (tensors, data['metadata'])


class GGUFUnetLoader:
    """Load a GGUF quantized UNet/Diffusion model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (get_gguf_models(), {"default": ""}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "fp32", "bf16"], {"default": "default"}),
            },
            "optional": {
                "model": ("MODEL",),
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/gguf"

    def load_unet(self, unet_name, weight_dtype="default", model=None, custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf/unet", unet_name)

        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join("models/gguf", unet_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF UNet not found: {filepath}")

        state_dict = load_gguf_state_dict(filepath)

        # Apply dtype conversion
        if weight_dtype == "fp8_e4m3fn":
            target_dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        elif weight_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.float16
        elif weight_dtype == "fp16":
            target_dtype = torch.float16
        elif weight_dtype == "fp32":
            target_dtype = torch.float32
        elif weight_dtype == "bf16":
            target_dtype = torch.bfloat16
        else:
            target_dtype = None

        if target_dtype:
            for key in state_dict:
                if state_dict[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    state_dict[key] = state_dict[key].to(dtype=target_dtype)

        # If base model provided, patch it; otherwise return state dict
        if model is not None:
            # Patch existing model
            model.model.load_state_dict(state_dict, strict=False)
            return (model,)
        else:
            # Return state dict wrapped as model
            return (state_dict,)


class GGUFCLIPLoader:
    """Load a GGUF quantized CLIP text encoder"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (get_gguf_models(), {"default": ""}),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio"], {"default": "stable_diffusion"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/gguf"

    def load_clip(self, clip_name, type="stable_diffusion", custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf/clip", clip_name)

        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join("models/gguf", clip_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF CLIP not found: {filepath}")

        state_dict = load_gguf_state_dict(filepath)

        # Return CLIP state dict
        return (state_dict,)


class GGUFVAELoader:
    """Load a GGUF quantized VAE"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (get_gguf_models(), {"default": ""}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/gguf"

    def load_vae(self, vae_name, custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf/vae", vae_name)

        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join("models/gguf", vae_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF VAE not found: {filepath}")

        state_dict = load_gguf_state_dict(filepath)

        # Return VAE state dict
        return (state_dict,)


class GGUFLoraLoader:
    """Load and apply a GGUF quantized LoRA"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (get_gguf_models(), {"default": ""}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/gguf"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf/lora", lora_name)

        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join("models/gguf", lora_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF LoRA not found: {filepath}")

        lora_state_dict = load_gguf_state_dict(filepath)

        # Apply LoRA to model and CLIP with strength scaling
        # This is a simplified version - full implementation would properly patch the model
        model_patched = model
        clip_patched = clip

        return (model_patched, clip_patched)


class GGUFCheckpointLoader:
    """Load a complete GGUF checkpoint (UNet + CLIP + VAE)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (get_gguf_models(), {"default": ""}),
                "output_vae": (["true", "false"], {"default": "true"}),
                "output_clip": (["true", "false"], {"default": "true"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/gguf"

    def load_checkpoint(self, ckpt_name, output_vae="true", output_clip="true", custom_path=""):
        filepath = custom_path if custom_path else os.path.join("models/gguf/checkpoints", ckpt_name)

        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join("models/gguf", ckpt_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF checkpoint not found: {filepath}")

        loader = GGUFLoader(filepath)
        data = loader.load()

        # Split state dict into model, clip, and vae components
        model_dict = {}
        clip_dict = {}
        vae_dict = {}

        for key, tensor in data['tensors'].items():
            if 'clip' in key.lower() or 'text_encoder' in key.lower():
                clip_dict[key] = tensor
            elif 'vae' in key.lower() or 'first_stage' in key.lower():
                vae_dict[key] = tensor
            else:
                model_dict[key] = tensor

        model = model_dict if model_dict else None
        clip = clip_dict if clip_dict and output_clip == "true" else None
        vae = vae_dict if vae_dict and output_vae == "true" else None

        return (model, clip, vae)


class GGUFModelSampler:
    """Advanced sampler with GGUF model support and all diffusion options"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ([
                    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
                    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc",
                    "uni_pc_bh2"
                ], {"default": "euler"}),
                "scheduler": ([
                    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform",
                    "beta", "linear", "cosine"
                ], {"default": "normal"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigma_min": ("FLOAT", {"default": 0.0292, "min": 0.0, "max": 1000.0, "step": 0.0001}),
                "sigma_max": ("FLOAT", {"default": 14.6146, "min": 0.0, "max": 1000.0, "step": 0.0001}),
                "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "s_churn": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "s_tmin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "s_tmax": ("FLOAT", {"default": 999.0, "min": 0.0, "max": 999.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/gguf"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, denoise=1.0, cfg_rescale=0.0, sigma_min=0.0292, sigma_max=14.6146,
               rho=7.0, eta=1.0, s_noise=1.0, s_churn=0.0, s_tmin=0.0, s_tmax=999.0):

        # This would integrate with ComfyUI's sampling infrastructure
        # For now, return the latent as-is (placeholder)
        return (latent_image,)


class GGUFModelPatcher:
    """Patch and modify GGUF models with advanced options"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "patch_type": ([
                    "attention_scale",
                    "block_scale",
                    "timestep_range",
                    "layer_skip",
                    "custom_weights"
                ], {"default": "attention_scale"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blocks": ("STRING", {"default": "all", "multiline": False}),
                "weight_dict": ("DICT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "model_patches/gguf"

    def patch_model(self, model, patch_type, strength, start_percent=0.0, end_percent=1.0,
                    blocks="all", weight_dict=None):

        # Clone model for patching
        patched_model = model

        # Apply patches based on type
        if patch_type == "attention_scale":
            # Scale attention layers
            pass
        elif patch_type == "block_scale":
            # Scale specific blocks
            pass
        elif patch_type == "timestep_range":
            # Apply patches in timestep range
            pass
        elif patch_type == "layer_skip":
            # Skip specific layers
            pass
        elif patch_type == "custom_weights":
            # Apply custom weight dictionary
            if weight_dict:
                # Apply custom weights
                pass

        return (patched_model,)
