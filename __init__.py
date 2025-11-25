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
    GGUFModelSaver,
    GGUFCheckpointSaver,
    GGUF5DTensorPatcher,
    GGUFTensorQuantizer,
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
    "GGUFModelSaver": GGUFModelSaver,
    "GGUFCheckpointSaver": GGUFCheckpointSaver,
    "GGUF5DTensorPatcher": GGUF5DTensorPatcher,
    "GGUFTensorQuantizer": GGUFTensorQuantizer,
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
    "GGUFModelSaver": "GGUF Model Saver",
    "GGUFCheckpointSaver": "GGUF Checkpoint Saver",
    "GGUF5DTensorPatcher": "GGUF 5D Tensor Patcher",
    "GGUFTensorQuantizer": "GGUF Tensor Quantizer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
