# ComfyUI GGUF Nodes

A comprehensive set of custom nodes for working with GGUF (GPT-Generated Unified Format) quantized models in ComfyUI. These nodes enable you to load and use quantized diffusion models, reducing memory usage while maintaining quality.

## Features

- **Full GGUF Support**: Load and save GGUF quantized models with proper dequantization
- **Complete Model Support**: UNet, CLIP, VAE, LoRA, and full checkpoints
- **Create GGUF Files**: Save models directly to GGUF format from ComfyUI
- **5D Tensor Support**: Full support for 5D tensors (batch, channels, depth, height, width)
- **Advanced Sampling**: Comprehensive sampler with all diffusion model options
- **Flexible Quantization**: Support for all quantization types (Q2_K through Q8_K, F16, F32)
- **Model Patching**: Advanced model modification and 5D tensor patching capabilities
- **Quantization-Aware Preprocessing**: Optimize tensors before quantization to preserve quality
- **Device Management**: Automatic device detection (CUDA, MPS, CPU) with manual override
- **Dtype Control**: Support for FP32, FP16, BF16, and FP8 formats
- **Quality Testing**: Preview quantization quality with detailed statistics

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

### 9. GGUF Model Saver
Save models to GGUF format with quantization (supports 2D-5D tensors).

**Inputs:**
- `model`: Model to save (MODEL, dict, or state_dict)
- `filename`: Output filename (e.g., "model.gguf")
- `quantization`: Quantization type (F32, F16, Q4_0-Q8_1, Q2_K-Q8_K)
- `save_path`: Save directory (models/gguf, models/gguf/unet, or custom)
- `custom_path`: Optional custom save path
- `metadata`: Optional metadata dictionary
- `apply_quantize_patch`: Apply quantization-aware patching (default: true)

**Outputs:**
- `filepath`: Path to saved GGUF file

**Features:**
- Supports 2D, 3D, 4D, and **5D tensors**
- Automatic quantization-aware patching to preserve quality
- All quantization types supported (Q2_K through Q8_K)
- Custom metadata support
- Automatic directory creation

### 10. GGUF Checkpoint Saver
Save complete checkpoints with separate quantization for each component.

**Inputs:**
- `filename`: Output filename
- `quantization_unet`: UNet quantization (default: Q4_K)
- `quantization_clip`: CLIP quantization (default: Q8_0)
- `quantization_vae`: VAE quantization (default: F16)
- `model`: UNet model (optional)
- `clip`: CLIP model (optional)
- `vae`: VAE model (optional)
- `save_path`: Save directory (default: models/gguf/checkpoints)
- `metadata`: Optional metadata dictionary

**Outputs:**
- `filepath`: Path to saved checkpoint

**Features:**
- Save UNet, CLIP, and VAE in single file
- Independent quantization per component
- Optimal quality/size balance (Q4_K for UNet, Q8_0 for CLIP, F16 for VAE)
- Automatic component prefixing

### 11. GGUF 5D Tensor Patcher
Apply advanced patching operations to 5D tensors before GGUF creation.

**Inputs:**
- `model`: Model with tensors to patch
- `patch_operation`: Operation type
- `target_quantization`: Target quantization for aware patching
- `scale_factor`: Scaling factor (0.1 to 10.0)
- `clip_min`: Minimum value for clipping (-100.0 to 0.0)
- `clip_max`: Maximum value for clipping (0.0 to 100.0)

**Patch Operations:**
- `normalize`: Zero-mean, unit-variance normalization
- `scale`: Multiply by scale factor
- `clip_range`: Clamp values to range
- `quantize_aware`: Adjust for target quantization type
- `reduce_dynamic_range`: Reduce dynamic range by 50%
- `adaptive_scale`: Scale based on tensor statistics

**Outputs:**
- `model`: Model with patched tensors

**Features:**
- **Full 5D tensor support** (batch, channels, depth, height, width)
- Dimension preservation
- Quantization-aware preprocessing
- Multiple patching strategies

**Use Cases:**
- Prepare models for low-bit quantization (Q2_K, Q3_K, Q4_K)
- Reduce quantization artifacts
- Normalize tensors for better compression
- Fine-tune dynamic range before saving

### 12. GGUF Tensor Quantizer
Test quantization on individual tensors with quality metrics.

**Inputs:**
- `tensor`: Tensor to quantize
- `quantization`: Target quantization type
- `apply_pre_quantization_patch`: Enable patching (default: true)
- `tensor_name`: Name for statistics output

**Outputs:**
- `quantized_tensor`: Quantized tensor
- `statistics`: Detailed quality metrics (string)

**Statistics Provided:**
- Original tensor stats (mean, std, min, max)
- Quantized tensor stats (mean, std, min, max)
- Error metrics (MSE, MAE, relative error)
- Tensor shape information

**Features:**
- Preview quantization quality before full model save
- Compare different quantization types
- Identify problematic layers
- Validate 5D tensor quantization

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

### Creating GGUF Files

1. **Save a Single Model:**
   - Load your model in ComfyUI
   - Add "GGUF Model Saver" node
   - Choose quantization type (Q4_K recommended)
   - Set filename and save path
   - Execute to save

2. **Save Complete Checkpoint:**
   - Load UNet, CLIP, and VAE models
   - Add "GGUF Checkpoint Saver" node
   - Set different quantization for each component
   - Save all in one file

3. **Optimize for Low-Bit Quantization:**
   - Load your model
   - Add "GGUF 5D Tensor Patcher" node
   - Select "quantize_aware" operation
   - Choose target quantization (e.g., Q4_K)
   - Connect to "GGUF Model Saver"

4. **Test Quantization Quality:**
   - Extract a tensor from your model
   - Add "GGUF Tensor Quantizer" node
   - Test different quantization types
   - Review statistics to find optimal setting
   - Apply to full model

### Working with 5D Tensors

**5D Tensor Support:**
- Dimensions: [batch, channels, depth, height, width]
- Fully supported in save/load operations
- All patching operations preserve 5D structure
- Automatic flattening and reshaping during quantization

**Example Workflow:**
1. Load model with 5D tensors
2. Apply "GGUF 5D Tensor Patcher" with "quantize_aware"
3. Save with "GGUF Model Saver" using Q4_K
4. Load back with "GGUF Model Loader"
5. Verify dimensions preserved

### Memory Optimization Tips

1. **Use Q4_K or Q5_K** for best quality/size balance
2. **Load UNet as GGUF** (largest component) and keep CLIP/VAE in FP16
3. **Use FP8** for UNet if your GPU supports it (RTX 40xx series)
4. **Set device to "cpu"** for CLIP to free up VRAM if needed

## Example Workflows

The `workflows/` directory contains 8 ready-to-use example workflows demonstrating all features:

### Quick Start Workflows
1. **01_basic_gguf_loading.json** - Load and use GGUF checkpoints
2. **02_create_gguf_model.json** - Convert models to GGUF format
3. **03_create_gguf_checkpoint.json** - Save complete optimized checkpoints

### Advanced Workflows
4. **04_5d_tensor_patching.json** - Optimize models with 5D tensor patching
5. **05_quantization_quality_test.json** - Test and compare quantization quality
6. **06_complete_optimization_workflow.json** - Full end-to-end pipeline

### Specialized Workflows
7. **07_load_individual_components.json** - Load UNet, CLIP, VAE separately
8. **08_gguf_lora_workflow.json** - Use GGUF LoRAs with quantized models

### How to Use Workflows

1. **Import into ComfyUI:**
   - Click "Load" in ComfyUI
   - Navigate to `custom_nodes/comfyui_gguf_marduk191/workflows/`
   - Select a workflow JSON file

2. **Configure:**
   - Update file paths to your models
   - Adjust quantization settings as needed
   - Modify prompts and parameters

3. **Execute:**
   - Click "Queue Prompt" to run
   - Monitor progress in ComfyUI
   - Check outputs

📚 **See [workflows/README.md](workflows/README.md) for detailed documentation of each workflow!**

## Creating and Converting Models to GGUF

### Direct Creation in ComfyUI (Recommended)

Use the built-in GGUF saver nodes:
- **GGUF Model Saver** - Save any model with chosen quantization
- **GGUF Checkpoint Saver** - Save complete checkpoints
- **GGUF 5D Tensor Patcher** - Optimize before saving
- **GGUF Tensor Quantizer** - Test quantization quality

### External Conversion Tools

For converting from other formats:
- `llama.cpp` convert scripts
- Custom conversion tools (community-developed)
- Note: Built-in savers are easier and support 5D tensors directly

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

### Version 1.1.0 (GGUF Creation & 5D Tensor Support)
- **NEW:** GGUF Model Saver - Save models to GGUF format
- **NEW:** GGUF Checkpoint Saver - Save complete checkpoints with component-specific quantization
- **NEW:** GGUF 5D Tensor Patcher - Advanced patching for 5D tensors
- **NEW:** GGUF Tensor Quantizer - Test and preview quantization quality
- **NEW:** Full 5D tensor support in all save/load operations
- **NEW:** Quantization-aware preprocessing for better quality
- **NEW:** GGUFWriter class with all quantization types
- **NEW:** Quality metrics and statistics for quantization testing
- **NEW:** 8 example workflows demonstrating all features
- 6 patching operations (normalize, scale, clip, quantize-aware, etc.)
- Complete GGUF creation pipeline in ComfyUI
- Improved documentation with creation workflows and workflow guide

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