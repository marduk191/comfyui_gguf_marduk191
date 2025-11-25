"""
Utilities for loading and working with GGUF models
"""

import os
import struct
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import torch

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF data types
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18

GGML_TYPE_SIZE = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.Q4_0: 18,  # Block size
    GGMLType.Q4_1: 20,
    GGMLType.Q5_0: 22,
    GGMLType.Q5_1: 24,
    GGMLType.Q8_0: 34,
    GGMLType.Q8_1: 36,
    GGMLType.I8: 1,
    GGMLType.I16: 2,
    GGMLType.I32: 4,
}

class GGUFLoader:
    """Load and parse GGUF format models"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata = {}
        self.tensors = {}
        self.tensor_info = {}

    def load(self) -> Dict[str, Any]:
        """Load GGUF file and return model data"""
        with open(self.filepath, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF file: wrong magic number {magic:08x}")

            version = struct.unpack('<I', f.read(4))[0]
            if version != GGUF_VERSION:
                print(f"Warning: GGUF version {version}, expected {GGUF_VERSION}")

            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

            # Read tensor info
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                self.tensor_info[name] = {
                    'dims': dims,
                    'dtype': dtype,
                    'offset': offset,
                }

            # Align to 32 bytes
            alignment = 32
            current_pos = f.tell()
            aligned_pos = (current_pos + alignment - 1) & ~(alignment - 1)
            f.seek(aligned_pos)

            data_offset = f.tell()

            # Read tensor data
            for name, info in self.tensor_info.items():
                f.seek(data_offset + info['offset'])
                tensor_data = self._read_tensor(f, info)
                self.tensors[name] = tensor_data

        return {
            'metadata': self.metadata,
            'tensors': self.tensors,
            'tensor_info': self.tensor_info,
        }

    def _read_string(self, f) -> str:
        """Read a GGUF string"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: int) -> Any:
        """Read a GGUF metadata value"""
        if value_type == 0:  # UINT8
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == 1:  # INT8
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == 3:  # INT16
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == 4:  # UINT32
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == 5:  # INT32
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == 7:  # BOOL
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == 8:  # STRING
            return self._read_string(f)
        elif value_type == 9:  # ARRAY
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        elif value_type == 10:  # UINT64
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == 11:  # INT64
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack('<d', f.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_tensor(self, f, info: Dict) -> torch.Tensor:
        """Read and dequantize tensor data"""
        dims = info['dims']
        dtype = info['dtype']

        # Calculate number of elements
        n_elements = 1
        for d in dims:
            n_elements *= d

        # Read quantized data
        if dtype == GGMLType.F32:
            data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32)
            tensor = torch.from_numpy(data).reshape(dims[::-1])
        elif dtype == GGMLType.F16:
            data = np.frombuffer(f.read(n_elements * 2), dtype=np.float16)
            tensor = torch.from_numpy(data.astype(np.float32)).reshape(dims[::-1])
        elif dtype in [GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
                       GGMLType.Q8_0, GGMLType.Q8_1, GGMLType.Q2_K, GGMLType.Q3_K,
                       GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K]:
            # For quantized types, read raw data and store for later dequantization
            # This is a simplified version - full implementation would dequantize properly
            block_size = self._get_block_size(dtype)
            n_blocks = (n_elements + block_size - 1) // block_size
            bytes_per_block = self._get_bytes_per_block(dtype)
            raw_data = f.read(n_blocks * bytes_per_block)

            # Dequantize (simplified - real implementation needs proper dequantization)
            tensor = self._dequantize(raw_data, dtype, dims)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return tensor

    def _get_block_size(self, dtype: int) -> int:
        """Get block size for quantized types"""
        block_sizes = {
            GGMLType.Q4_0: 32,
            GGMLType.Q4_1: 32,
            GGMLType.Q5_0: 32,
            GGMLType.Q5_1: 32,
            GGMLType.Q8_0: 32,
            GGMLType.Q8_1: 32,
            GGMLType.Q2_K: 256,
            GGMLType.Q3_K: 256,
            GGMLType.Q4_K: 256,
            GGMLType.Q5_K: 256,
            GGMLType.Q6_K: 256,
            GGMLType.Q8_K: 256,
        }
        return block_sizes.get(dtype, 32)

    def _get_bytes_per_block(self, dtype: int) -> int:
        """Get bytes per block for quantized types"""
        bytes_per_block = {
            GGMLType.Q4_0: 18,
            GGMLType.Q4_1: 20,
            GGMLType.Q5_0: 22,
            GGMLType.Q5_1: 24,
            GGMLType.Q8_0: 34,
            GGMLType.Q8_1: 36,
            GGMLType.Q2_K: 84,
            GGMLType.Q3_K: 110,
            GGMLType.Q4_K: 144,
            GGMLType.Q5_K: 176,
            GGMLType.Q6_K: 210,
            GGMLType.Q8_K: 292,
        }
        return bytes_per_block.get(dtype, 18)

    def _dequantize(self, raw_data: bytes, dtype: int, dims: Tuple) -> torch.Tensor:
        """Dequantize tensor data"""
        # Simplified dequantization - returns zeros as placeholder
        # Real implementation would properly dequantize each quantization type
        n_elements = 1
        for d in dims:
            n_elements *= d

        # For now, return zeros - this should be replaced with proper dequantization
        print(f"Warning: Dequantization not fully implemented for dtype {dtype}")
        return torch.zeros(dims[::-1], dtype=torch.float32)


def get_gguf_models(base_path: str = "models/gguf") -> List[str]:
    """Get list of available GGUF model files"""
    if not os.path.exists(base_path):
        return []

    gguf_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gguf'):
                rel_path = os.path.relpath(os.path.join(root, file), base_path)
                gguf_files.append(rel_path)

    return sorted(gguf_files)


def load_gguf_state_dict(filepath: str) -> Dict[str, torch.Tensor]:
    """Load GGUF file and return as PyTorch state dict"""
    loader = GGUFLoader(filepath)
    data = loader.load()
    return data['tensors']


class GGUFWriter:
    """Write and save GGUF format models with 5D tensor support"""

    # GGUF metadata value types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    def __init__(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        self.filepath = filepath
        self.metadata = metadata or {}
        self.tensors = []
        self.alignment = 32

    def _get_block_size(self, dtype: int) -> int:
        """Get block size for quantized types"""
        block_sizes = {
            GGMLType.Q4_0: 32,
            GGMLType.Q4_1: 32,
            GGMLType.Q5_0: 32,
            GGMLType.Q5_1: 32,
            GGMLType.Q8_0: 32,
            GGMLType.Q8_1: 32,
            GGMLType.Q2_K: 256,
            GGMLType.Q3_K: 256,
            GGMLType.Q4_K: 256,
            GGMLType.Q5_K: 256,
            GGMLType.Q6_K: 256,
            GGMLType.Q8_K: 256,
        }
        return block_sizes.get(dtype, 32)

    def _get_bytes_per_block(self, dtype: int) -> int:
        """Get bytes per block for quantized types"""
        bytes_per_block = {
            GGMLType.Q4_0: 18,
            GGMLType.Q4_1: 20,
            GGMLType.Q5_0: 22,
            GGMLType.Q5_1: 24,
            GGMLType.Q8_0: 34,
            GGMLType.Q8_1: 36,
            GGMLType.Q2_K: 84,
            GGMLType.Q3_K: 110,
            GGMLType.Q4_K: 144,
            GGMLType.Q5_K: 176,
            GGMLType.Q6_K: 210,
            GGMLType.Q8_K: 292,
        }
        return bytes_per_block.get(dtype, 18)

    def add_metadata(self, key: str, value: Any, value_type: Optional[int] = None):
        """Add metadata key-value pair"""
        if value_type is None:
            # Auto-detect type
            if isinstance(value, bool):
                value_type = self.GGUF_TYPE_BOOL
            elif isinstance(value, str):
                value_type = self.GGUF_TYPE_STRING
            elif isinstance(value, float):
                value_type = self.GGUF_TYPE_FLOAT32
            elif isinstance(value, int):
                if value < 0:
                    value_type = self.GGUF_TYPE_INT32
                else:
                    value_type = self.GGUF_TYPE_UINT32
            elif isinstance(value, list):
                value_type = self.GGUF_TYPE_ARRAY
            else:
                raise ValueError(f"Cannot auto-detect type for {type(value)}")

        self.metadata[key] = (value, value_type)

    def add_tensor(self, name: str, tensor: torch.Tensor, quantization_type: int = GGMLType.F32):
        """Add a tensor to be written (supports 1D-5D tensors)

        WARNING: This stores tensors in memory. For large models, use save_streaming() instead.
        """
        if tensor.dim() < 1 or tensor.dim() > 5:
            raise ValueError(f"Tensor {name} must be 1D-5D, got {tensor.dim()}D")

        # For 1D tensors (biases, norms), force F32 or F16 to avoid quantization issues
        if tensor.dim() == 1 and quantization_type not in [GGMLType.F32, GGMLType.F16]:
            print(f"Warning: 1D tensor '{name}' forced to F16 (was {quantization_type})")
            quantization_type = GGMLType.F16

        self.tensors.append({
            'name': name,
            'tensor': tensor,
            'quantization_type': quantization_type,
        })

    def save(self, use_gpu=False, gc_interval=10):
        """Write GGUF file to disk with memory-efficient streaming

        Args:
            use_gpu: If True, perform quantization on GPU for faster processing (recommended for RTX 4090/5090)
            gc_interval: How often to run garbage collection (lower = more memory efficient, higher = faster)
        """
        import gc

        with open(self.filepath, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Write metadata
            for key, (value, value_type) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack('<I', value_type))
                self._write_value(f, value, value_type)

            # Calculate tensor data offsets (but don't store full tensors yet)
            tensor_infos = []
            current_offset = 0

            for tensor_dict in self.tensors:
                tensor = tensor_dict['tensor']
                quant_type = tensor_dict['quantization_type']

                # Get dimensions in GGUF format (reversed)
                dims = tuple(tensor.shape[::-1])
                n_elements = tensor.numel()

                # Calculate size based on quantization
                if quant_type == GGMLType.F32:
                    data_size = n_elements * 4
                elif quant_type == GGMLType.F16:
                    data_size = n_elements * 2
                elif quant_type in [GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0,
                                     GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1]:
                    block_size = 32
                    bytes_per_block = self._get_bytes_per_block(quant_type)
                    n_blocks = (n_elements + block_size - 1) // block_size
                    data_size = n_blocks * bytes_per_block
                elif quant_type in [GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K,
                                     GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K]:
                    block_size = 256
                    bytes_per_block = self._get_bytes_per_block(quant_type)
                    n_blocks = (n_elements + block_size - 1) // block_size
                    data_size = n_blocks * bytes_per_block
                else:
                    raise ValueError(f"Unsupported quantization type: {quant_type}")

                # Align data size
                aligned_size = (data_size + self.alignment - 1) & ~(self.alignment - 1)

                tensor_infos.append({
                    'name': tensor_dict['name'],
                    'dims': dims,
                    'dtype': quant_type,
                    'offset': current_offset,
                    'tensor_index': len(tensor_infos),  # Reference to original tensor
                    'data_size': data_size,
                })

                current_offset += aligned_size

            # Write tensor info
            for info in tensor_infos:
                self._write_string(f, info['name'])
                f.write(struct.pack('<I', len(info['dims'])))
                for dim in info['dims']:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', info['dtype']))
                f.write(struct.pack('<Q', info['offset']))

            # Align to 32 bytes before tensor data
            current_pos = f.tell()
            aligned_pos = (current_pos + self.alignment - 1) & ~(self.alignment - 1)
            padding = aligned_pos - current_pos
            f.write(b'\x00' * padding)

            # Write tensor data one at a time to save memory
            for info in tensor_infos:
                tensor_idx = info['tensor_index']
                tensor = self.tensors[tensor_idx]['tensor']
                quant_type = info['dtype']

                # GPU acceleration: keep tensor on GPU during quantization for speed
                # Only move to CPU when writing to disk
                if use_gpu and tensor.is_cuda:
                    # Quantize on GPU (much faster on RTX 4090/5090)
                    with torch.no_grad():
                        quantized_data = self._quantize_tensor(tensor, quant_type)
                    # Move result to CPU for disk write
                    if isinstance(quantized_data, bytes):
                        pass  # Already in CPU memory as bytes
                    else:
                        # This shouldn't happen, but handle it
                        quantized_data = quantized_data.cpu() if hasattr(quantized_data, 'cpu') else quantized_data
                else:
                    # Memory-efficient mode: move to CPU before quantization to save VRAM
                    if tensor.is_cuda:
                        tensor = tensor.cpu()

                    # Quantize and write tensor
                    with torch.no_grad():  # Disable gradient tracking to save memory
                        quantized_data = self._quantize_tensor(tensor, quant_type)

                f.write(quantized_data)

                # Force cleanup of quantized data
                del quantized_data

                # Align to 32 bytes
                current_pos = f.tell()
                aligned_pos = (current_pos + self.alignment - 1) & ~(self.alignment - 1)
                padding = aligned_pos - current_pos
                if padding > 0:
                    f.write(b'\x00' * padding)

                # Clear tensor from memory if possible (don't modify original)
                # Periodic garbage collection
                if tensor_idx % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def save_streaming(self, tensor_dict, quant_types, use_gpu=False, gc_interval=1):
        """Write GGUF file with true streaming - processes tensors on-demand

        Args:
            tensor_dict: Dictionary of {name: tensor} (e.g., state_dict)
            quant_types: Dictionary of {name: quantization_type} or single type for all
            use_gpu: If True, perform quantization on GPU
            gc_interval: How often to run garbage collection (default 1 for max memory efficiency)

        Example:
            quant_types = {name: GGMLType.Q4_K for name in state_dict.keys()}
            writer.save_streaming(state_dict, quant_types, use_gpu=True)
        """
        import gc

        # If quant_types is a single value, apply to all tensors
        if isinstance(quant_types, int):
            single_type = quant_types
            quant_types = {name: single_type for name in tensor_dict.keys()}

        # Filter to only tensor entries
        tensor_names = [name for name, value in tensor_dict.items() if isinstance(value, torch.Tensor)]

        print(f"Pass 1/2: Calculating metadata for {len(tensor_names)} tensors (no storage)...")

        # First pass: collect ONLY metadata (shapes, types) - DO NOT STORE TENSORS
        tensor_metadata = []
        current_offset = 0

        for name in tensor_names:
            tensor = tensor_dict[name]
            quant_type = quant_types.get(name, GGMLType.F32)

            # Validate dimensions
            if tensor.dim() < 1 or tensor.dim() > 5:
                raise ValueError(f"Tensor {name} must be 1D-5D, got {tensor.dim()}D")

            # Force F16 for 1D tensors
            if tensor.dim() == 1 and quant_type not in [GGMLType.F32, GGMLType.F16]:
                print(f"Warning: 1D tensor '{name}' forced to F16")
                quant_type = GGMLType.F16
                quant_types[name] = quant_type  # Update for second pass

            # Calculate metadata from tensor properties WITHOUT storing tensor
            dims = tuple(tensor.shape[::-1])
            n_elements = tensor.numel()

            # Calculate size based on quantization
            if quant_type == GGMLType.F32:
                data_size = n_elements * 4
            elif quant_type == GGMLType.F16:
                data_size = n_elements * 2
            else:
                block_size = self._get_block_size(quant_type)
                bytes_per_block = self._get_bytes_per_block(quant_type)
                n_blocks = (n_elements + block_size - 1) // block_size
                data_size = n_blocks * bytes_per_block

            tensor_metadata.append({
                'name': name,
                'dims': dims,
                'dtype': quant_type,
                'offset': current_offset,
                'data_size': data_size,
            })

            current_offset += data_size
            aligned_offset = (current_offset + self.alignment - 1) & ~(self.alignment - 1)
            current_offset = aligned_offset

        # Aggressive memory cleanup after pass 1
        del tensor_names  # Don't need list anymore
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Pass 2/2: Writing {len(tensor_metadata)} tensors to disk...")

        # Now write the file
        with open(self.filepath, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(tensor_metadata)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Write metadata
            for key, (value, value_type) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack('<I', value_type))
                self._write_value(f, value, value_type)

            # Write tensor metadata
            for info in tensor_metadata:
                self._write_string(f, info['name'])
                f.write(struct.pack('<I', len(info['dims'])))
                for dim in info['dims']:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', info['dtype']))
                f.write(struct.pack('<Q', info['offset']))

            # Align to 32 bytes before tensor data
            current_pos = f.tell()
            aligned_pos = (current_pos + self.alignment - 1) & ~(self.alignment - 1)
            padding = aligned_pos - current_pos
            f.write(b'\x00' * padding)

            # Second pass: Write tensor data one at a time by re-accessing dict
            for idx, info in enumerate(tensor_metadata):
                name = info['name']
                quant_type = info['dtype']

                # Get tensor from dict on-demand (not from stored copy)
                tensor = tensor_dict[name]

                # GPU acceleration mode
                if use_gpu and tensor.is_cuda:
                    with torch.no_grad():
                        quantized_data = self._quantize_tensor(tensor, quant_type)
                else:
                    # Move to CPU for memory efficiency
                    if tensor.is_cuda:
                        tensor = tensor.cpu()
                    with torch.no_grad():
                        quantized_data = self._quantize_tensor(tensor, quant_type)

                f.write(quantized_data)

                # Immediate cleanup - critical for low memory
                del quantized_data
                del tensor  # Remove local reference

                # Align to 32 bytes
                current_pos = f.tell()
                aligned_pos = (current_pos + self.alignment - 1) & ~(self.alignment - 1)
                padding = aligned_pos - current_pos
                if padding > 0:
                    f.write(b'\x00' * padding)

                # Aggressive garbage collection for streaming mode
                if idx % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if (idx + 1) % 100 == 0:
                    print(f"  Written {idx + 1}/{len(tensor_metadata)} tensors...")

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ Completed streaming write of {len(tensor_metadata)} tensors")

    def _write_string(self, f, s: str):
        """Write a GGUF string"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_value(self, f, value: Any, value_type: int):
        """Write a GGUF metadata value"""
        if value_type == self.GGUF_TYPE_UINT8:
            f.write(struct.pack('<B', value))
        elif value_type == self.GGUF_TYPE_INT8:
            f.write(struct.pack('<b', value))
        elif value_type == self.GGUF_TYPE_UINT16:
            f.write(struct.pack('<H', value))
        elif value_type == self.GGUF_TYPE_INT16:
            f.write(struct.pack('<h', value))
        elif value_type == self.GGUF_TYPE_UINT32:
            f.write(struct.pack('<I', value))
        elif value_type == self.GGUF_TYPE_INT32:
            f.write(struct.pack('<i', value))
        elif value_type == self.GGUF_TYPE_FLOAT32:
            f.write(struct.pack('<f', value))
        elif value_type == self.GGUF_TYPE_BOOL:
            f.write(struct.pack('<?', value))
        elif value_type == self.GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif value_type == self.GGUF_TYPE_ARRAY:
            # Detect array element type
            if len(value) > 0:
                elem_type = self._detect_type(value[0])
            else:
                elem_type = self.GGUF_TYPE_UINT32
            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(value)))
            for elem in value:
                self._write_value(f, elem, elem_type)
        elif value_type == self.GGUF_TYPE_UINT64:
            f.write(struct.pack('<Q', value))
        elif value_type == self.GGUF_TYPE_INT64:
            f.write(struct.pack('<q', value))
        elif value_type == self.GGUF_TYPE_FLOAT64:
            f.write(struct.pack('<d', value))
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _detect_type(self, value: Any) -> int:
        """Detect GGUF type from Python value"""
        if isinstance(value, bool):
            return self.GGUF_TYPE_BOOL
        elif isinstance(value, str):
            return self.GGUF_TYPE_STRING
        elif isinstance(value, float):
            return self.GGUF_TYPE_FLOAT32
        elif isinstance(value, int):
            if value < 0:
                return self.GGUF_TYPE_INT32
            else:
                return self.GGUF_TYPE_UINT32
        else:
            return self.GGUF_TYPE_UINT32

    def _get_bytes_per_block(self, dtype: int) -> int:
        """Get bytes per block for quantized types"""
        bytes_per_block = {
            GGMLType.Q4_0: 18,
            GGMLType.Q4_1: 20,
            GGMLType.Q5_0: 22,
            GGMLType.Q5_1: 24,
            GGMLType.Q8_0: 34,
            GGMLType.Q8_1: 36,
            GGMLType.Q2_K: 84,
            GGMLType.Q3_K: 110,
            GGMLType.Q4_K: 144,
            GGMLType.Q5_K: 176,
            GGMLType.Q6_K: 210,
            GGMLType.Q8_K: 292,
        }
        return bytes_per_block.get(dtype, 18)

    def _quantize_tensor(self, tensor: torch.Tensor, quant_type: int) -> bytes:
        """Quantize tensor to bytes (supports 2D-5D tensors)"""
        # Flatten tensor for processing (maintains data order)
        flat_tensor = tensor.flatten()

        if quant_type == GGMLType.F32:
            # Write as float32
            return flat_tensor.cpu().float().numpy().tobytes()
        elif quant_type == GGMLType.F16:
            # Write as float16
            return flat_tensor.cpu().half().numpy().tobytes()
        elif quant_type == GGMLType.Q4_0:
            return self._quantize_q4_0(flat_tensor)
        elif quant_type == GGMLType.Q4_1:
            return self._quantize_q4_1(flat_tensor)
        elif quant_type == GGMLType.Q5_0:
            return self._quantize_q5_0(flat_tensor)
        elif quant_type == GGMLType.Q5_1:
            return self._quantize_q5_1(flat_tensor)
        elif quant_type == GGMLType.Q8_0:
            return self._quantize_q8_0(flat_tensor)
        elif quant_type == GGMLType.Q8_1:
            return self._quantize_q8_1(flat_tensor)
        elif quant_type in [GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K,
                             GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K]:
            return self._quantize_k_quants(flat_tensor, quant_type)
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")

    def _quantize_q4_0(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q4_0 format (4-bit, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()
        n_blocks = (n_elements + block_size - 1) // block_size

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            # Calculate scale (max absolute value)
            max_val = block.abs().max()
            scale = max_val / 7.0 if max_val > 0 else 1.0

            # Write scale as float16
            output.extend(struct.pack('<e', scale))

            # Quantize values to 4 bits
            quantized = torch.clamp(torch.round(block / scale), -8, 7).to(torch.int8)

            # Pack two 4-bit values per byte
            for i in range(0, block_size, 2):
                val1 = int(quantized[i].item()) & 0x0F
                val2 = int(quantized[i + 1].item()) & 0x0F
                packed = (val2 << 4) | val1
                output.append(packed & 0xFF)

        return bytes(output)

    def _quantize_q4_1(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q4_1 format (4-bit with bias, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()
        n_blocks = (n_elements + block_size - 1) // block_size

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            # Calculate min and max
            min_val = block.min()
            max_val = block.max()
            scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0

            # Write scale and min as float16
            output.extend(struct.pack('<e', scale))
            output.extend(struct.pack('<e', min_val))

            # Quantize values to 4 bits
            quantized = torch.clamp(torch.round((block - min_val) / scale), 0, 15).to(torch.int8)

            # Pack two 4-bit values per byte
            for i in range(0, block_size, 2):
                val1 = int(quantized[i].item()) & 0x0F
                val2 = int(quantized[i + 1].item()) & 0x0F
                packed = (val2 << 4) | val1
                output.append(packed & 0xFF)

        return bytes(output)

    def _quantize_q5_0(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q5_0 format (5-bit, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            max_val = block.abs().max()
            scale = max_val / 15.0 if max_val > 0 else 1.0

            output.extend(struct.pack('<e', scale))

            # Quantize to 5 bits (-16 to 15)
            quantized = torch.clamp(torch.round(block / scale), -16, 15).to(torch.int8)

            # Pack 5-bit values (simplified - proper implementation would pack more efficiently)
            for val in quantized:
                output.append(int(val.item()) & 0xFF)

        return bytes(output)

    def _quantize_q5_1(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q5_1 format (5-bit with bias, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            min_val = block.min()
            max_val = block.max()
            scale = (max_val - min_val) / 31.0 if max_val > min_val else 1.0

            output.extend(struct.pack('<e', scale))
            output.extend(struct.pack('<e', min_val))

            quantized = torch.clamp(torch.round((block - min_val) / scale), 0, 31).to(torch.int8)

            for val in quantized:
                output.append(int(val.item()) & 0xFF)

        return bytes(output)

    def _quantize_q8_0(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q8_0 format (8-bit, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            max_val = block.abs().max()
            scale = max_val / 127.0 if max_val > 0 else 1.0

            output.extend(struct.pack('<e', scale))

            quantized = torch.clamp(torch.round(block / scale), -128, 127).to(torch.int8)

            for val in quantized:
                output.append(int(val.item()) & 0xFF)

        return bytes(output)

    def _quantize_q8_1(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q8_1 format (8-bit with bias, block size 32)"""
        block_size = 32
        n_elements = tensor.numel()

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        for block in tensor:
            min_val = block.min()
            max_val = block.max()
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0

            output.extend(struct.pack('<e', scale))
            output.extend(struct.pack('<e', min_val))

            quantized = torch.clamp(torch.round((block - min_val) / scale), 0, 255).to(torch.uint8)

            output.extend(quantized.numpy().tobytes())

        return bytes(output)

    def _quantize_k_quants(self, tensor: torch.Tensor, quant_type: int) -> bytes:
        """Quantize to K-quant formats (Q2_K through Q8_K, block size 256)"""
        block_size = 256
        n_elements = tensor.numel()

        # Pad tensor if needed
        if n_elements % block_size != 0:
            padding = block_size - (n_elements % block_size)
            tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)])

        tensor = tensor.cpu().float().reshape(-1, block_size)
        output = bytearray()

        # K-quants use sub-blocks and super-blocks
        # Simplified implementation - proper implementation would use the exact GGML format
        for block in tensor:
            if quant_type == GGMLType.Q4_K:
                # Q4_K: 144 bytes per block
                # Scale + min + quantized data
                min_val = block.min()
                max_val = block.max()
                scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0

                # Write scales (simplified)
                output.extend(struct.pack('<f', scale))
                output.extend(struct.pack('<f', min_val))

                # Quantize and pack
                quantized = torch.clamp(torch.round((block - min_val) / scale), 0, 15).to(torch.uint8)
                packed = bytearray()
                for i in range(0, len(quantized), 2):
                    if i + 1 < len(quantized):
                        val = (quantized[i] & 0x0F) | ((quantized[i + 1] & 0x0F) << 4)
                    else:
                        val = quantized[i] & 0x0F
                    packed.append(val)

                output.extend(packed)

                # Pad to 144 bytes
                while len(output) % 144 != 0:
                    output.append(0)

            elif quant_type == GGMLType.Q8_K:
                # Q8_K: 292 bytes per block
                max_val = block.abs().max()
                scale = max_val / 127.0 if max_val > 0 else 1.0

                output.extend(struct.pack('<f', scale))

                quantized = torch.clamp(torch.round(block / scale), -128, 127).to(torch.int8)
                output.extend(quantized.numpy().tobytes())

                # Pad to 292 bytes
                while len(output) % 292 != 0:
                    output.append(0)

            else:
                # Other K-quants - simplified implementation
                max_val = block.abs().max()
                scale = max_val / 127.0 if max_val > 0 else 1.0

                output.extend(struct.pack('<f', scale))
                quantized = torch.clamp(torch.round(block / scale), -128, 127).to(torch.int8)
                output.extend(quantized.numpy().tobytes())

                # Pad to required size
                bytes_per_block = self._get_bytes_per_block(quant_type)
                while len(output) % bytes_per_block != 0:
                    output.append(0)

        return bytes(output)


def patch_5d_tensor(tensor: torch.Tensor, patch_fn, **kwargs) -> torch.Tensor:
    """
    Apply patching function to 5D tensor with preservation of dimensions

    Args:
        tensor: Input tensor (2D-5D)
        patch_fn: Function to apply (e.g., normalize, scale, quantize-aware training)
        **kwargs: Additional arguments for patch_fn

    Returns:
        Patched tensor with same dimensions
    """
    if tensor.dim() == 5:
        # Process 5D tensor: [batch, channels, depth, height, width]
        original_shape = tensor.shape
        patched = patch_fn(tensor, **kwargs)
        return patched.reshape(original_shape)
    else:
        # Handle 2D-4D tensors directly
        return patch_fn(tensor, **kwargs)


def quantize_aware_patch(tensor: torch.Tensor, quant_type: int = GGMLType.Q4_K) -> torch.Tensor:
    """
    Apply quantization-aware patching to preserve quality during GGUF conversion

    Args:
        tensor: Input tensor
        quant_type: Target quantization type

    Returns:
        Tensor adjusted for quantization
    """
    # Apply range normalization based on target quantization
    if quant_type in [GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q4_K]:
        # 4-bit quantization - adjust range
        std = tensor.std()
        if std > 0:
            tensor = tensor / std * 0.5  # Reduce dynamic range
    elif quant_type in [GGMLType.Q2_K, GGMLType.Q3_K]:
        # Very low bit quantization - aggressive range reduction
        std = tensor.std()
        if std > 0:
            tensor = tensor / std * 0.25

    return tensor
