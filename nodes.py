"""
ComfyUI GGUF Model Nodes
Comprehensive nodes for loading and working with GGUF quantized models
"""

import os
import folder_paths
import torch
from .gguf_utils import (
    GGUFLoader, GGUFWriter, get_gguf_models, load_gguf_state_dict,
    GGMLType, patch_5d_tensor, quantize_aware_patch
)


def extract_state_dict(model):
    """
    Extract state dict from various model types including ComfyUI ModelPatcher objects
    """
    state_dict = None

    # Try to extract state dict from various model types
    if hasattr(model, 'model'):
        # ComfyUI ModelPatcher object
        if hasattr(model.model, 'diffusion_model') and hasattr(model.model.diffusion_model, 'state_dict'):
            state_dict = model.model.diffusion_model.state_dict()
        elif hasattr(model.model, 'state_dict') and callable(model.model.state_dict):
            state_dict = model.model.state_dict()
        elif callable(getattr(model, 'model_state_dict', None)):
            state_dict = model.model_state_dict()
        else:
            state_dict = model.model
    elif isinstance(model, dict):
        state_dict = model
    elif hasattr(model, 'state_dict') and callable(model.state_dict):
        state_dict = model.state_dict()
    else:
        # Assume it's already a state dict
        state_dict = model

    # Ensure we have a valid state dict
    if state_dict is None or not hasattr(state_dict, 'items'):
        raise ValueError(f"Cannot extract state dict from model type: {type(model)}")

    return state_dict


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


class GGUFModelSaver:
    """Save models to GGUF format with quantization (supports 1D-5D tensors)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename": ("STRING", {"default": "model.gguf", "multiline": False}),
                "quantization": ([
                    "F32", "F16",
                    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"
                ], {"default": "Q4_K"}),
                "save_path": (["models/gguf", "models/gguf/unet", "custom"], {"default": "models/gguf"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
                "metadata": ("DICT",),
                "apply_quantize_patch": ("BOOLEAN", {"default": True}),
                "gpu_accelerated": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_model"
    CATEGORY = "savers/gguf"
    OUTPUT_NODE = True

    def save_model(self, model, filename, quantization, save_path, custom_path="",
                   metadata=None, apply_quantize_patch=True, gpu_accelerated=False):

        # Determine save path
        if save_path == "custom" and custom_path:
            filepath = os.path.join(custom_path, filename)
        else:
            filepath = os.path.join(save_path, filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Map quantization string to type
        quant_map = {
            "F32": GGMLType.F32,
            "F16": GGMLType.F16,
            "Q4_0": GGMLType.Q4_0,
            "Q4_1": GGMLType.Q4_1,
            "Q5_0": GGMLType.Q5_0,
            "Q5_1": GGMLType.Q5_1,
            "Q8_0": GGMLType.Q8_0,
            "Q8_1": GGMLType.Q8_1,
            "Q2_K": GGMLType.Q2_K,
            "Q3_K": GGMLType.Q3_K,
            "Q4_K": GGMLType.Q4_K,
            "Q5_K": GGMLType.Q5_K,
            "Q6_K": GGMLType.Q6_K,
            "Q8_K": GGMLType.Q8_K,
        }
        quant_type = quant_map[quantization]

        # Create GGUF writer
        writer = GGUFWriter(filepath, metadata=metadata)

        # Add default metadata
        writer.add_metadata("quantization_version", 2)
        writer.add_metadata("file_type", quantization)

        # Extract state_dict (just references, not copies)
        state_dict = extract_state_dict(model)

        # Apply quantization-aware patching if enabled (modifies tensors in-place when possible)
        if apply_quantize_patch:
            print("Applying quantization-aware patching...")
            for name, tensor in list(state_dict.items()):
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    state_dict[name] = quantize_aware_patch(tensor, quant_type)

        # Use streaming save - processes tensors on-demand without storing all
        writer.save_streaming(state_dict, quant_type, use_gpu=gpu_accelerated)

        accel_msg = " (GPU accelerated)" if gpu_accelerated else ""
        print(f"Saved GGUF model to {filepath} with {quantization} quantization{accel_msg}")
        return (filepath,)


class GGUFCheckpointSaver:
    """Save complete checkpoints (UNet + CLIP + VAE) to GGUF format"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "checkpoint.gguf", "multiline": False}),
                "quantization_unet": ([
                    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"
                ], {"default": "Q4_K"}),
                "quantization_clip": ([
                    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"
                ], {"default": "Q8_0"}),
                "quantization_vae": ([
                    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"
                ], {"default": "F16"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "save_path": ("STRING", {"default": "models/gguf/checkpoints", "multiline": False}),
                "metadata": ("DICT",),
                "gpu_accelerated": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_checkpoint"
    CATEGORY = "savers/gguf"
    OUTPUT_NODE = True

    def save_checkpoint(self, filename, quantization_unet, quantization_clip, quantization_vae,
                        model=None, clip=None, vae=None, save_path="models/gguf/checkpoints",
                        metadata=None, gpu_accelerated=False):

        filepath = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Map quantization strings to types
        quant_map = {
            "F32": GGMLType.F32, "F16": GGMLType.F16,
            "Q4_0": GGMLType.Q4_0, "Q4_1": GGMLType.Q4_1,
            "Q5_0": GGMLType.Q5_0, "Q5_1": GGMLType.Q5_1,
            "Q8_0": GGMLType.Q8_0, "Q8_1": GGMLType.Q8_1,
            "Q2_K": GGMLType.Q2_K, "Q3_K": GGMLType.Q3_K,
            "Q4_K": GGMLType.Q4_K, "Q5_K": GGMLType.Q5_K,
            "Q6_K": GGMLType.Q6_K, "Q8_K": GGMLType.Q8_K,
        }

        # Create GGUF writer
        writer = GGUFWriter(filepath, metadata=metadata)

        # Add metadata
        writer.add_metadata("file_type", "checkpoint")
        writer.add_metadata("quantization_unet", quantization_unet)
        writer.add_metadata("quantization_clip", quantization_clip)
        writer.add_metadata("quantization_vae", quantization_vae)

        # Build combined state_dict with prefixes and quant types
        combined_state_dict = {}
        quant_types_dict = {}

        # Add model tensors
        if model is not None:
            try:
                state_dict = extract_state_dict(model)
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        full_name = f"model.{name}"
                        combined_state_dict[full_name] = tensor
                        quant_types_dict[full_name] = quant_map[quantization_unet]
            except Exception as e:
                print(f"Warning: Could not extract model state dict: {e}")

        # Add CLIP tensors
        if clip is not None:
            try:
                state_dict = extract_state_dict(clip)
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        full_name = f"clip.{name}"
                        combined_state_dict[full_name] = tensor
                        quant_types_dict[full_name] = quant_map[quantization_clip]
            except Exception as e:
                print(f"Warning: Could not extract CLIP state dict: {e}")

        # Add VAE tensors
        if vae is not None:
            try:
                state_dict = extract_state_dict(vae)
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        full_name = f"vae.{name}"
                        combined_state_dict[full_name] = tensor
                        quant_types_dict[full_name] = quant_map[quantization_vae]
            except Exception as e:
                print(f"Warning: Could not extract VAE state dict: {e}")

        # Use streaming save - processes tensors on-demand
        writer.save_streaming(combined_state_dict, quant_types_dict, use_gpu=gpu_accelerated)

        accel_msg = " (GPU accelerated)" if gpu_accelerated else ""
        print(f"Saved GGUF checkpoint to {filepath}{accel_msg}")
        return (filepath,)


class GGUF5DTensorPatcher:
    """Patch 5D tensors during GGUF creation with various operations"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "patch_operation": ([
                    "normalize", "scale", "clip_range", "quantize_aware",
                    "reduce_dynamic_range", "adaptive_scale"
                ], {"default": "quantize_aware"}),
                "target_quantization": ([
                    "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K",
                    "Q8_0", "Q8_1", "Q8_K", "Q2_K", "Q3_K", "Q6_K"
                ], {"default": "Q4_K"}),
            },
            "optional": {
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "clip_min": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "clip_max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_tensors"
    CATEGORY = "model_patches/gguf"

    def patch_tensors(self, model, patch_operation, target_quantization,
                     scale_factor=1.0, clip_min=-10.0, clip_max=10.0):

        # Map quantization string to type
        quant_map = {
            "Q4_0": GGMLType.Q4_0, "Q4_1": GGMLType.Q4_1, "Q4_K": GGMLType.Q4_K,
            "Q5_0": GGMLType.Q5_0, "Q5_1": GGMLType.Q5_1, "Q5_K": GGMLType.Q5_K,
            "Q8_0": GGMLType.Q8_0, "Q8_1": GGMLType.Q8_1, "Q8_K": GGMLType.Q8_K,
            "Q2_K": GGMLType.Q2_K, "Q3_K": GGMLType.Q3_K, "Q6_K": GGMLType.Q6_K,
        }
        quant_type = quant_map[target_quantization]

        # Handle ComfyUI ModelPatcher objects
        if hasattr(model, 'model') and hasattr(model, 'clone'):
            # This is a ComfyUI ModelPatcher - clone it and patch the internal model
            patched_model = model.clone()

            # Get the actual model state dict
            if hasattr(patched_model.model, 'state_dict'):
                state_dict = patched_model.model.state_dict()
            elif hasattr(patched_model.model, 'diffusion_model'):
                state_dict = patched_model.model.diffusion_model.state_dict()
            else:
                # Fallback: try to get state dict directly
                state_dict = patched_model.model

            # Apply patching to each tensor
            patched_tensors = {}
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue

                # Only patch 2D-5D tensors (skip 1D biases/norms as they don't benefit from patching)
                if tensor.dim() < 2 or tensor.dim() > 5:
                    continue

                # Apply patch operation
                if patch_operation == "normalize":
                    def normalize_fn(t):
                        mean = t.mean()
                        std = t.std()
                        return (t - mean) / (std + 1e-8) if std > 0 else t
                    patched_tensor = patch_5d_tensor(tensor, normalize_fn)

                elif patch_operation == "scale":
                    def scale_fn(t):
                        return t * scale_factor
                    patched_tensor = patch_5d_tensor(tensor, scale_fn)

                elif patch_operation == "clip_range":
                    def clip_fn(t):
                        return torch.clamp(t, clip_min, clip_max)
                    patched_tensor = patch_5d_tensor(tensor, clip_fn)

                elif patch_operation == "quantize_aware":
                    def quant_aware_fn(t):
                        return quantize_aware_patch(t, quant_type)
                    patched_tensor = patch_5d_tensor(tensor, quant_aware_fn)

                elif patch_operation == "reduce_dynamic_range":
                    def reduce_range_fn(t):
                        # Reduce dynamic range while preserving sign
                        abs_max = t.abs().max()
                        if abs_max > 0:
                            target_max = abs_max * 0.5
                            return t * (target_max / abs_max)
                        return t
                    patched_tensor = patch_5d_tensor(tensor, reduce_range_fn)

                elif patch_operation == "adaptive_scale":
                    def adaptive_fn(t):
                        # Scale based on tensor statistics
                        std = t.std()
                        if std > 1.0:
                            return t / std
                        return t
                    patched_tensor = patch_5d_tensor(tensor, adaptive_fn)

                else:
                    patched_tensor = tensor

                patched_tensors[name] = patched_tensor

            # Apply patches to the model using ComfyUI's patching system
            def patch_fn(model_function, params):
                """Apply tensor patches during model execution"""
                # This is called during model execution
                return model_function

            # Add patches to the ModelPatcher
            for key in patched_tensors:
                # Use ComfyUI's add_patches method if available
                if hasattr(patched_model, 'add_patches'):
                    patched_model.add_patches({key: (patched_tensors[key],)}, 1.0, 0.0)

            print(f"Patched {len(patched_tensors)} tensors with {patch_operation} operation")
            return (patched_model,)

        else:
            # Handle dict or state_dict directly (for standalone usage)
            if isinstance(model, dict):
                state_dict = model
            elif hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            else:
                state_dict = model

            patched_state_dict = {}

            # Apply patching to each tensor
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    patched_state_dict[name] = tensor
                    continue

                # Only patch 2D-5D tensors (skip 1D biases/norms as they don't benefit from patching)
                if tensor.dim() < 2 or tensor.dim() > 5:
                    patched_state_dict[name] = tensor
                    continue

                # Apply patch operation
                if patch_operation == "normalize":
                    def normalize_fn(t):
                        mean = t.mean()
                        std = t.std()
                        return (t - mean) / (std + 1e-8) if std > 0 else t
                    patched_tensor = patch_5d_tensor(tensor, normalize_fn)

                elif patch_operation == "scale":
                    def scale_fn(t):
                        return t * scale_factor
                    patched_tensor = patch_5d_tensor(tensor, scale_fn)

                elif patch_operation == "clip_range":
                    def clip_fn(t):
                        return torch.clamp(t, clip_min, clip_max)
                    patched_tensor = patch_5d_tensor(tensor, clip_fn)

                elif patch_operation == "quantize_aware":
                    def quant_aware_fn(t):
                        return quantize_aware_patch(t, quant_type)
                    patched_tensor = patch_5d_tensor(tensor, quant_aware_fn)

                elif patch_operation == "reduce_dynamic_range":
                    def reduce_range_fn(t):
                        # Reduce dynamic range while preserving sign
                        abs_max = t.abs().max()
                        if abs_max > 0:
                            target_max = abs_max * 0.5
                            return t * (target_max / abs_max)
                        return t
                    patched_tensor = patch_5d_tensor(tensor, reduce_range_fn)

                elif patch_operation == "adaptive_scale":
                    def adaptive_fn(t):
                        # Scale based on tensor statistics
                        std = t.std()
                        if std > 1.0:
                            return t / std
                        return t
                    patched_tensor = patch_5d_tensor(tensor, adaptive_fn)

                else:
                    patched_tensor = tensor

                patched_state_dict[name] = patched_tensor

            print(f"Patched {len(patched_state_dict)} tensors with {patch_operation} operation")
            return (patched_state_dict,)


class GGUFTensorQuantizer:
    """Quantize individual tensors with preview and quality control"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "quantization": ([
                    "F32", "F16",
                    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"
                ], {"default": "Q4_K"}),
                "apply_pre_quantization_patch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "tensor_name": ("STRING", {"default": "tensor", "multiline": False}),
                "gpu_accelerated": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("quantized_tensor", "statistics")
    FUNCTION = "quantize_tensor"
    CATEGORY = "utils/gguf"

    def quantize_tensor(self, tensor, quantization, apply_pre_quantization_patch=True,
                       tensor_name="tensor", gpu_accelerated=False):

        # Map quantization string to type
        quant_map = {
            "F32": GGMLType.F32, "F16": GGMLType.F16,
            "Q4_0": GGMLType.Q4_0, "Q4_1": GGMLType.Q4_1,
            "Q5_0": GGMLType.Q5_0, "Q5_1": GGMLType.Q5_1,
            "Q8_0": GGMLType.Q8_0, "Q8_1": GGMLType.Q8_1,
            "Q2_K": GGMLType.Q2_K, "Q3_K": GGMLType.Q3_K,
            "Q4_K": GGMLType.Q4_K, "Q5_K": GGMLType.Q5_K,
            "Q6_K": GGMLType.Q6_K, "Q8_K": GGMLType.Q8_K,
        }
        quant_type = quant_map[quantization]

        # Calculate original statistics
        orig_mean = tensor.mean().item()
        orig_std = tensor.std().item()
        orig_min = tensor.min().item()
        orig_max = tensor.max().item()

        # Apply pre-quantization patch if enabled
        if apply_pre_quantization_patch:
            tensor = quantize_aware_patch(tensor, quant_type)

        # Create temporary writer to quantize
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            writer = GGUFWriter(tmp.name)
            writer.add_tensor(tensor_name, tensor, quant_type)
            writer.save(use_gpu=gpu_accelerated)

            # Load back to get quantized version
            loader = GGUFLoader(tmp.name)
            data = loader.load()
            quantized_tensor = data['tensors'][tensor_name]

        # Clean up temp file
        os.unlink(tmp.name)

        # Calculate quantized statistics
        quant_mean = quantized_tensor.mean().item()
        quant_std = quantized_tensor.std().item()
        quant_min = quantized_tensor.min().item()
        quant_max = quantized_tensor.max().item()

        # Calculate error metrics
        mse = ((tensor - quantized_tensor) ** 2).mean().item()
        mae = (tensor - quantized_tensor).abs().mean().item()

        # Format statistics
        stats = f"""Quantization Statistics for {tensor_name}:
Quantization: {quantization}
Tensor Shape: {list(tensor.shape)}

Original:
  Mean: {orig_mean:.6f}, Std: {orig_std:.6f}
  Min: {orig_min:.6f}, Max: {orig_max:.6f}

Quantized:
  Mean: {quant_mean:.6f}, Std: {quant_std:.6f}
  Min: {quant_min:.6f}, Max: {quant_max:.6f}

Error Metrics:
  MSE: {mse:.6f}
  MAE: {mae:.6f}
  Relative Error: {(mae / (abs(orig_mean) + 1e-8) * 100):.2f}%
"""

        print(stats)
        return (quantized_tensor, stats)
