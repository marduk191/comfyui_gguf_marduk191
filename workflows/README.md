# GGUF Workflow Examples

This directory contains example workflows demonstrating all features of the ComfyUI GGUF nodes.

## 📋 Workflow Index

### Basic Workflows

#### 01_basic_gguf_loading.json
**Purpose:** Load and use a GGUF checkpoint for image generation

**Features Demonstrated:**
- GGUFCheckpointLoader node
- Basic image generation pipeline
- Loading all components (UNet, CLIP, VAE) from single file

**Use Case:** Quick start with GGUF models

**Nodes Used:**
- GGUFCheckpointLoader
- CLIPTextEncode (x2)
- KSampler
- VAEDecode
- SaveImage

---

#### 07_load_individual_components.json
**Purpose:** Load UNet, CLIP, and VAE separately

**Features Demonstrated:**
- GGUFUnetLoader node
- GGUFCLIPLoader node
- GGUFVAELoader node
- Component-specific loading
- Mixed quantization levels

**Use Case:** Fine-grained control over model loading, mix different quantization levels

**Nodes Used:**
- GGUFUnetLoader
- GGUFCLIPLoader
- GGUFVAELoader
- CLIPTextEncode (x2)
- KSampler
- VAEDecode
- SaveImage

---

### GGUF Creation Workflows

#### 02_create_gguf_model.json
**Purpose:** Convert a standard model to GGUF format

**Features Demonstrated:**
- GGUFModelSaver node
- Single model quantization
- Q4_K quantization (recommended)
- File path output

**Use Case:** Create GGUF files from existing models

**Nodes Used:**
- CheckpointLoaderSimple
- GGUFModelSaver
- ShowText

**Settings:**
- Quantization: Q4_K (4-bit, recommended)
- Apply quantization-aware patching: Enabled
- Output: 40-60% size reduction

---

#### 03_create_gguf_checkpoint.json
**Purpose:** Save complete checkpoint with optimal component quantization

**Features Demonstrated:**
- GGUFCheckpointSaver node
- Component-specific quantization
- Optimal quantization strategy
- Single-file checkpoint creation

**Use Case:** Create distribution-ready GGUF checkpoints

**Nodes Used:**
- CheckpointLoaderSimple
- GGUFCheckpointSaver
- ShowText

**Recommended Settings:**
- UNet: Q4_K (maximum compression for largest component)
- CLIP: Q8_0 (preserve text understanding)
- VAE: F16 (maintain image quality)

**Expected Results:**
- File size: 40-50% of original
- Quality: Near-identical to original
- Perfect for distribution and storage

---

### Advanced Optimization Workflows

#### 04_5d_tensor_patching.json
**Purpose:** Optimize models for quantization using 5D tensor patching

**Features Demonstrated:**
- GGUF5DTensorPatcher node
- Quantization-aware preprocessing
- 6 patching operations
- 5D tensor support

**Use Case:** Prepare models for low-bit quantization (Q2_K, Q3_K, Q4_K)

**Nodes Used:**
- CheckpointLoaderSimple
- GGUF5DTensorPatcher
- GGUFModelSaver
- ShowText

**Patching Operations Available:**
1. **quantize_aware** (Recommended): Optimizes tensors for target quantization
   - Adjusts dynamic range
   - Reduces quantization artifacts
   - Best for Q2_K through Q5_K

2. **normalize**: Zero-mean, unit-variance normalization
   - Good for unnormalized models

3. **scale**: Multiply by scale_factor
   - Fine-tune weight magnitudes

4. **clip_range**: Clamp to [clip_min, clip_max]
   - Remove outliers

5. **reduce_dynamic_range**: 50% range reduction
   - For aggressive compression

6. **adaptive_scale**: Statistics-based scaling
   - Automatic normalization

**Best Practice:** Always use `quantize_aware` before quantizing to Q4_K or lower!

---

#### 05_quantization_quality_test.json
**Purpose:** Test and compare different quantization types

**Features Demonstrated:**
- GGUFTensorQuantizer node
- Quality metrics (MSE, MAE, relative error)
- Side-by-side comparison
- Statistical analysis

**Use Case:** Find optimal quantization level for your model

**Nodes Used:**
- CheckpointLoaderSimple
- ModelToTensor (extract test tensor)
- GGUFTensorQuantizer (x3 for Q4_K, Q5_K, Q8_K)
- ShowText (x3 for statistics)

**Metrics Displayed:**
- Original tensor statistics
- Quantized tensor statistics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Relative Error percentage

**Quality Guidelines:**
| Quantization | Typical Error | Quality | Recommendation |
|--------------|---------------|---------|----------------|
| Q8_K | < 1% | Best | High-quality archival |
| Q6_K | < 2% | Excellent | Production use |
| Q5_K | < 3% | Very Good | Recommended |
| Q4_K | < 5% | Good | **Most popular** |
| Q3_K | < 10% | Acceptable | Experimental |
| Q2_K | > 10% | Poor | Use with caution |

---

#### 06_complete_optimization_workflow.json
**Purpose:** End-to-end optimization and generation pipeline

**Features Demonstrated:**
- Complete GGUF workflow
- Optimization → Save → Load → Generate
- All major nodes in one workflow
- Real-world usage pattern

**Use Case:** Production pipeline for model optimization and deployment

**Workflow Steps:**
1. Load original model (any format)
2. Apply quantization-aware patching
3. Save as optimized GGUF checkpoint
4. Load back GGUF checkpoint
5. Generate image using quantized model
6. Compare results

**Nodes Used:**
- CheckpointLoaderSimple
- GGUF5DTensorPatcher
- GGUFCheckpointSaver
- GGUFCheckpointLoader
- CLIPTextEncode (x2)
- EmptyLatentImage
- KSampler
- VAEDecode
- SaveImage

**Benefits:**
- 40-50% size reduction
- Near-identical quality
- Reduced VRAM usage
- Faster loading
- Ready for distribution

---

### Specialized Workflows

#### 08_gguf_lora_workflow.json
**Purpose:** Use GGUF-quantized LoRA with GGUF base model

**Features Demonstrated:**
- GGUFLoraLoader node
- Independent strength control
- Quantized LoRA application
- Model + CLIP patching

**Use Case:** Apply style/character LoRAs with quantized models

**Nodes Used:**
- GGUFCheckpointLoader
- GGUFLoraLoader
- CLIPTextEncode (x2)
- EmptyLatentImage
- KSampler
- VAEDecode
- SaveImage

**LoRA Strength Parameters:**
- **strength_model** (0.0 to 20.0):
  - 1.0 = normal strength
  - < 1.0 = subtle effect
  - \> 1.0 = amplified effect
  - Negative values = inverse effect

- **strength_clip** (0.0 to 20.0):
  - Independent CLIP control
  - Affects prompt understanding
  - Usually keep around 1.0

**LoRA Quantization Recommendations:**
- Q8_0: Best quality (recommended)
- Q4_K: Good compression
- F16: No quantization

**Advantages:**
- Smaller LoRA files
- Fast loading
- Stack multiple quantized LoRAs
- Full compatibility with GGUF base models

---

## 🚀 Quick Start Guide

### 1. Import a Workflow
1. Open ComfyUI
2. Click "Load" button
3. Navigate to `custom_nodes/comfyui_gguf_marduk191/workflows/`
4. Select a workflow JSON file
5. Click "Open"

### 2. Configure Paths
- Update file paths to point to your models
- Ensure GGUF files are in correct directories:
  - `models/gguf/checkpoints/` for complete checkpoints
  - `models/gguf/unet/` for UNet models
  - `models/gguf/clip/` for CLIP models
  - `models/gguf/vae/` for VAE models
  - `models/gguf/lora/` for LoRA models

### 3. Execute Workflow
- Click "Queue Prompt" to run the workflow
- Monitor progress in the ComfyUI interface
- Check output for saved files or generated images

---

## 📊 Workflow Selection Guide

### I want to...

**...start using GGUF models:**
→ Use `01_basic_gguf_loading.json`

**...create my first GGUF file:**
→ Use `02_create_gguf_model.json`

**...save a complete checkpoint:**
→ Use `03_create_gguf_checkpoint.json`

**...optimize for Q4_K quantization:**
→ Use `04_5d_tensor_patching.json`

**...test quantization quality:**
→ Use `05_quantization_quality_test.json`

**...see the complete process:**
→ Use `06_complete_optimization_workflow.json`

**...load components separately:**
→ Use `07_load_individual_components.json`

**...use LoRA with GGUF:**
→ Use `08_gguf_lora_workflow.json`

---

## ⚙️ Configuration Tips

### Quantization Selection

**For Storage/Distribution:**
- UNet: Q4_K (best compression)
- CLIP: Q8_0 (preserve quality)
- VAE: F16 (no artifacts)

**For High Quality:**
- UNet: Q5_K or Q6_K
- CLIP: F16
- VAE: F16

**For Maximum Compression:**
- UNet: Q3_K or Q4_K
- CLIP: Q4_K or Q8_0
- VAE: Q4_K or F16

### Memory Management

**Low VRAM (< 8GB):**
1. Use Q4_K for UNet
2. Load CLIP on CPU
3. Use standard VAE (not GGUF)

**Medium VRAM (8-12GB):**
1. Use Q4_K or Q5_K for UNet
2. Use Q8_0 for CLIP
3. Use F16 for VAE

**High VRAM (> 12GB):**
1. Use Q6_K or Q8_K for UNet
2. Use F16 for CLIP
3. Use F16 for VAE

---

## 🔧 Troubleshooting

### Workflow won't load
- Ensure GGUF nodes are installed
- Restart ComfyUI after installing nodes
- Check for missing dependencies

### File not found errors
- Verify file paths match your directory structure
- Check files have `.gguf` extension
- Ensure files are in correct subdirectories

### Quality issues
- Try higher quantization (Q5_K, Q6_K, Q8_K)
- Enable quantization-aware patching
- Test with quality testing workflow first

### Out of memory
- Use lower quantization (Q4_K)
- Reduce batch size
- Load CLIP on CPU
- Close other applications

---

## 📝 Customization

All workflows can be customized by:
1. Changing quantization types
2. Adjusting sampler parameters
3. Modifying prompt text
4. Changing image dimensions
5. Adding/removing nodes

Feel free to use these as templates for your own workflows!

---

## 🎯 Best Practices

1. **Always test quantization quality first** using workflow 05
2. **Use quantization-aware patching** for Q4_K and below (workflow 04)
3. **Save original models** before converting to GGUF
4. **Start with Q4_K** - best quality/size balance
5. **Keep VAE as F16** to maintain image quality
6. **Backup important models** before experimenting

---

## 📚 Additional Resources

- [Main README](../README.md) - Full documentation
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)

---

## 💡 Tips & Tricks

### Batch Processing
Create variants of these workflows to process multiple models:
1. Load different models
2. Use same quantization settings
3. Queue multiple prompts

### Workflow Sharing
Export your customized workflows:
1. Save workflow in ComfyUI
2. Export as JSON
3. Share with community

### Performance Optimization
For faster processing:
1. Use Q4_K for maximum speed
2. Enable GPU quantization if supported
3. Reduce image dimensions for testing

---

## 🤝 Contributing Workflows

Have a useful workflow? Consider contributing!
1. Create a descriptive workflow
2. Add inline notes explaining steps
3. Test thoroughly
4. Submit a pull request

---

## Version Information

These workflows are designed for:
- ComfyUI GGUF Nodes v1.1.0+
- ComfyUI (latest version)
- GGUF Format v3

Last Updated: 2025-11-25
