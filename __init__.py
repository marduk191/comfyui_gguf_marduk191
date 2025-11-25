"""
ComfyUI GGUF Nodes
A comprehensive set of nodes for working with GGUF quantized models in ComfyUI
"""

from .nodes import (
    GGUFModelLoader,
    GGUFUnetLoader,
    GGUFCLIPLoader,
    GGUFVAELoader,
    GGUFLoraLoader,
    GGUFCheckpointLoader,
    GGUFModelSampler,
    GGUFModelPatcher,
)

NODE_CLASS_MAPPINGS = {
    "GGUFModelLoader": GGUFModelLoader,
    "GGUFUnetLoader": GGUFUnetLoader,
    "GGUFCLIPLoader": GGUFCLIPLoader,
    "GGUFVAELoader": GGUFVAELoader,
    "GGUFLoraLoader": GGUFLoraLoader,
    "GGUFCheckpointLoader": GGUFCheckpointLoader,
    "GGUFModelSampler": GGUFModelSampler,
    "GGUFModelPatcher": GGUFModelPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFModelLoader": "GGUF Model Loader",
    "GGUFUnetLoader": "GGUF UNet Loader",
    "GGUFCLIPLoader": "GGUF CLIP Loader",
    "GGUFVAELoader": "GGUF VAE Loader",
    "GGUFLoraLoader": "GGUF LoRA Loader",
    "GGUFCheckpointLoader": "GGUF Checkpoint Loader",
    "GGUFModelSampler": "GGUF Model Sampler",
    "GGUFModelPatcher": "GGUF Model Patcher",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
