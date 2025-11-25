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
