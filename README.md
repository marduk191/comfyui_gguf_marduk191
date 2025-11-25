# ComfyUI GGUF Nodes

A comprehensive set of custom nodes for working with GGUF (GPT-Generated Unified Format) quantized models in ComfyUI. These nodes enable you to load and use quantized diffusion models, reducing memory usage while maintaining quality.

## Features

- **Full GGUF Support**: Load GGUF quantized models with proper dequantization
- **Complete Model Support**: UNet, CLIP, VAE, LoRA, and full checkpoints
- **Advanced Sampling**: Comprehensive sampler with all diffusion model options
- **Flexible Loading**: Support for multiple quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
- **Model Patching**: Advanced model modification and patching capabilities
- **Device Management**: Automatic device detection (CUDA, MPS, CPU) with manual override
- **Dtype Control**: Support for FP32, FP16, BF16, and FP8 formats

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/comfyui_gguf_marduk191.git
```

2. Install dependencies:
```bash
cd comfyui_gguf_marduk191
pip install -r requirements.txt
```

3. Restart ComfyUI

## Directory Structure

Create the following directories in your ComfyUI models folder for GGUF models:

```
ComfyUI/models/
├── gguf/
│   ├── checkpoints/    # Full model checkpoints
│   ├── unet/          # UNet/Diffusion models
│   ├── clip/          # CLIP text encoders
│   ├── vae/           # VAE models
│   └── lora/          # LoRA adapters
```

## Available Nodes

### 1. GGUF Model Loader
Generic loader for any GGUF model file.

**Inputs:**
- `model_path`: Path to GGUF file (dropdown of available models)
- `device`: Target device (auto, cpu, cuda, mps)
- `dtype`: Data type (auto, float32, float16, bfloat16)
- `custom_path`: Optional custom file path

**Outputs:**
- `model`: Loaded model tensors
- `metadata`: GGUF metadata dictionary

### 2. GGUF UNet Loader
Load GGUF quantized UNet/Diffusion models.

**Inputs:**
- `unet_name`: UNet model file
- `weight_dtype`: Weight precision (default, fp8_e4m3fn, fp8_e5m2, fp16, fp32, bf16)
- `model`: Optional base model to patch
- `custom_path`: Optional custom file path

**Outputs:**
- `model`: Loaded/patched UNet model

**Features:**
- Supports FP8 quantization for even lower memory usage
- Can patch existing models or load standalone
- Automatic dtype conversion

### 3. GGUF CLIP Loader
Load GGUF quantized CLIP text encoders.

**Inputs:**
- `clip_name`: CLIP model file
- `type`: Model type (stable_diffusion, stable_cascade, sd3, stable_audio)
- `custom_path`: Optional custom file path

**Outputs:**
- `clip`: Loaded CLIP model

**Supported Types:**
- Stable Diffusion 1.x/2.x
- Stable Cascade
- Stable Diffusion 3
- Stable Audio

### 4. GGUF VAE Loader
Load GGUF quantized VAE models.

**Inputs:**
- `vae_name`: VAE model file
- `custom_path`: Optional custom file path

**Outputs:**
- `vae`: Loaded VAE model

### 5. GGUF LoRA Loader
Load and apply GGUF quantized LoRA adapters.

**Inputs:**
- `model`: Base model to apply LoRA to
- `clip`: Base CLIP model
- `lora_name`: LoRA file
- `strength_model`: Model strength (0.0 to 20.0)
- `strength_clip`: CLIP strength (0.0 to 20.0)
- `custom_path`: Optional custom file path

**Outputs:**
- `model`: Model with LoRA applied
- `clip`: CLIP with LoRA applied

**Features:**
- Independent strength control for model and CLIP
- Supports negative strengths for inverse effects
- High precision strength steps (0.01)

### 6. GGUF Checkpoint Loader
Load complete GGUF checkpoints containing UNet, CLIP, and VAE.

**Inputs:**
- `ckpt_name`: Checkpoint file
- `output_vae`: Enable VAE output (true/false)
- `output_clip`: Enable CLIP output (true/false)
- `custom_path`: Optional custom file path

**Outputs:**
- `model`: UNet model
- `clip`: CLIP text encoder
- `vae`: VAE

**Features:**
- Automatic component detection and separation
- Selective output (skip VAE or CLIP if not needed)
- Single file convenience

### 7. GGUF Model Sampler
Advanced sampler with comprehensive diffusion model options.

**Required Inputs:**
- `model`: Model to sample from
- `seed`: Random seed (0 to 2^64-1)
- `steps`: Number of sampling steps (1 to 10000)
- `cfg`: Classifier-free guidance scale (0.0 to 100.0)
- `sampler_name`: Sampling algorithm
- `scheduler`: Noise schedule
- `positive`: Positive conditioning
- `negative`: Negative conditioning
- `latent_image`: Input latent
- `denoise`: Denoising strength (0.0 to 1.0)

**Optional Inputs:**
- `cfg_rescale`: CFG rescale factor (0.0 to 1.0)
- `sigma_min`: Minimum noise level (0.0 to 1000.0)
- `sigma_max`: Maximum noise level (0.0 to 1000.0)
- `rho`: Karras scheduler rho parameter (0.0 to 100.0)
- `eta`: DDIM/ancestral eta parameter (0.0 to 100.0)
- `s_noise`: Noise multiplier (0.0 to 100.0)
- `s_churn`: Stochastic churn (0.0 to 100.0)
- `s_tmin`: Minimum timestep for churn (0.0 to 100.0)
- `s_tmax`: Maximum timestep for churn (0.0 to 999.0)

**Supported Samplers:**
- euler, euler_ancestral
- heun, heunpp2
- dpm_2, dpm_2_ancestral
- lms
- dpm_fast, dpm_adaptive
- dpmpp_2s_ancestral
- dpmpp_sde, dpmpp_sde_gpu
- dpmpp_2m, dpmpp_2m_sde, dpmpp_2m_sde_gpu
- dpmpp_3m_sde, dpmpp_3m_sde_gpu
- ddpm, ddim
- lcm
- uni_pc, uni_pc_bh2

**Supported Schedulers:**
- normal
- karras
- exponential
- sgm_uniform
- simple
- ddim_uniform
- beta
- linear
- cosine

**Outputs:**
- `latent`: Sampled latent image

### 8. GGUF Model Patcher
Advanced model patching and modification.

**Inputs:**
- `model`: Model to patch
- `patch_type`: Type of patch to apply
- `strength`: Patch strength (0.0 to 10.0)
- `start_percent`: Start of effect range (0.0 to 1.0)
- `end_percent`: End of effect range (0.0 to 1.0)
- `blocks`: Target blocks ("all" or specific blocks)
- `weight_dict`: Custom weight dictionary

**Patch Types:**
- `attention_scale`: Scale attention layer outputs
- `block_scale`: Scale specific block outputs
- `timestep_range`: Apply patches in timestep range
- `layer_skip`: Skip specific layers
- `custom_weights`: Apply custom weight modifications

**Outputs:**
- `model`: Patched model

## Quantization Types Supported

- **Q4_0**: 4-bit quantization, original
- **Q4_1**: 4-bit quantization, improved
- **Q5_0**: 5-bit quantization
- **Q5_1**: 5-bit quantization, improved
- **Q8_0**: 8-bit quantization
- **Q8_1**: 8-bit quantization, improved
- **Q2_K**: 2-bit k-quantization
- **Q3_K**: 3-bit k-quantization
- **Q4_K**: 4-bit k-quantization (recommended)
- **Q5_K**: 5-bit k-quantization
- **Q6_K**: 6-bit k-quantization (high quality)
- **Q8_K**: 8-bit k-quantization (highest quality)
- **F16**: 16-bit floating point
- **F32**: 32-bit floating point

## Usage Examples

### Basic Workflow

1. **Load a GGUF Checkpoint:**
   - Add "GGUF Checkpoint Loader" node
   - Select your GGUF checkpoint file
   - Connect outputs to your workflow

2. **Use Individual Components:**
   - Add "GGUF UNet Loader" for the diffusion model
   - Add "GGUF CLIP Loader" for the text encoder
   - Add "GGUF VAE Loader" for the VAE
   - Connect to standard ComfyUI nodes

3. **Apply LoRA:**
   - Load base model with GGUF loaders
   - Add "GGUF LoRA Loader"
   - Set strength values
   - Connect to sampling pipeline

4. **Advanced Sampling:**
   - Use "GGUF Model Sampler" for fine-grained control
   - Adjust sigma values for custom noise schedules
   - Experiment with different samplers and schedulers

### Memory Optimization Tips

1. **Use Q4_K or Q5_K** for best quality/size balance
2. **Load UNet as GGUF** (largest component) and keep CLIP/VAE in FP16
3. **Use FP8** for UNet if your GPU supports it (RTX 40xx series)
4. **Set device to "cpu"** for CLIP to free up VRAM if needed

## Converting Models to GGUF

To convert existing models to GGUF format, use tools like:
- `llama.cpp` convert scripts
- Custom conversion tools (community-developed)

## Troubleshooting

### Model Not Loading
- Ensure GGUF file is in correct directory
- Check file isn't corrupted
- Verify GGUF format version is 3

### Out of Memory
- Try lower quantization (Q4_K instead of Q8_K)
- Use FP8 for UNet
- Move CLIP to CPU
- Reduce batch size

### Quality Issues
- Use higher quantization (Q6_K or Q8_K)
- Check if model was properly quantized
- Verify source model quality

## Technical Details

### GGUF Format
GGUF is a binary format for storing neural network models with:
- Efficient quantization
- Metadata storage
- Fast loading
- Cross-platform compatibility

### Dequantization
Models are dequantized on-the-fly during loading, converting quantized weights back to floating point for inference.

### Performance
- **Memory**: 40-80% reduction vs FP16
- **Speed**: Slight overhead during loading, minimal during inference
- **Quality**: Q6_K and above near-identical to FP16

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Credits

- GGUF format: llama.cpp project
- ComfyUI: comfyanonymous
- Author: marduk191

## Support

For issues, questions, or suggestions:
- GitHub Issues: https://github.com/marduk191/comfyui_gguf_marduk191/issues
- ComfyUI Discord: #custom-nodes

## Changelog

### Version 1.0.0 (Initial Release)
- GGUF Model Loader
- GGUF UNet Loader with FP8 support
- GGUF CLIP Loader
- GGUF VAE Loader
- GGUF LoRA Loader
- GGUF Checkpoint Loader
- GGUF Model Sampler with all diffusion options
- GGUF Model Patcher
- Support for all major quantization types
- Automatic device detection
- Comprehensive dtype support