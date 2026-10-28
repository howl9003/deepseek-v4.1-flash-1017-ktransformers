"""
Weight loaders for different formats.

This module provides loaders for:
- SafeTensor format (for AMX quantized weights)
- GGUF format (for Llamafile quantized weights)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from enum import IntEnum
from safetensors import safe_open
import gguf.constants as _gguf_constants
import gguf.gguf_reader as _gguf_reader
from gguf.gguf_reader import GGUFReader

# ggml types newer than the installed gguf-py: id -> (name, (block_size, type_size)).
# MXFP4 is upstream 39, Q2_0 (2-bit {-1,0,1,2}, fp16 scale per 64) is upstream 42.
_EXTRA_GGML_TYPES_BY_ID = {
    39: ("MXFP4", (32, 1 + 16)),
    42: ("Q2_0", (64, 2 + 64 // 4)),
}
_EXTRA_GGML_QUANT_SIZES = {name: sizes for name, sizes in _EXTRA_GGML_TYPES_BY_ID.values()}


class _ExtraGGMLType:
    """Stand-in for a GGMLQuantizationType member the installed gguf-py lacks."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        try:
            return int(other) == self.value
        except (TypeError, ValueError):
            return NotImplemented

    def __repr__(self):
        return f"<GGMLQuantizationType.{self.name}: {self.value}>"


def _install_extra_ggml_types():
    """Let gguf-py's reader accept tensor types it was released without."""
    enum_cls = _gguf_constants.GGMLQuantizationType
    for type_id, (name, sizes) in _EXTRA_GGML_TYPES_BY_ID.items():
        member = getattr(enum_cls, name, None)
        if member is not None:
            _gguf_constants.GGML_QUANT_SIZES.setdefault(member, sizes)
    if all(hasattr(enum_cls, name) for name, _ in _EXTRA_GGML_TYPES_BY_ID.values()):
        return

    class _Lookup:
        def __call__(self, value):
            try:
                return enum_cls(value)
            except ValueError:
                extra = _EXTRA_GGML_TYPES_BY_ID.get(int(value))
                if extra is None:
                    raise
                member = _ExtraGGMLType(int(value), extra[0])
                _gguf_constants.GGML_QUANT_SIZES[member] = extra[1]
                return member

        def __getattr__(self, key):
            return getattr(enum_cls, key)

    _gguf_reader.GGMLQuantizationType = _Lookup()


_install_extra_ggml_types()


class GGMLQuantizationType(IntEnum):
    """GGML quantization type enumeration"""

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
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    MXFP4 = 39
    Q2_0 = 42


def translate_name_to_gguf(name):
    """
    Translate PyTorch tensor name to GGUF format
    """
    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")
    name = name.replace(".self_attn.q_norm", ".attn_q_norm")
    name = name.replace(".self_attn.k_norm", ".attn_k_norm")
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".exp_probs_b.bias")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")
    name = name.replace(".mlp.experts", "")
    name = name.replace(".mlp.experts.ffn_down_exps", ".ffn_down_exps")
    name = name.replace(".mlp.experts.ffn_gate_exps", ".ffn_gate_exps")
    name = name.replace(".mlp.experts.ffn_up_exps", ".ffn_up_exps")
    name = name.replace(".block_sparse_moe.gate.", ".ffn_gate_inp.")
    name = name.replace(".block_sparse_moe.experts", "")
    name = name.replace(".feed_forward.experts", "")
    name = name.replace(".feed_forward.router", ".ffn_gate_inp")
    name = name.replace(".feed_forward.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".feed_forward.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".feed_forward.shared_experts.up_proj", ".ffn_up_shexp")
    return name


class SafeTensorLoader:
    """
    SafeTensor format loader for AMX quantized weights.

    Supports loading tensors from .safetensors files with NUMA-sharded expert weights.
    """

    tensor_file_map: dict
    tensor_type_map: dict
    file_handle_map: dict
    tensor_device_map: dict

    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path
        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def close_all_handles(self):
        """Close all file handles and clear the handle map.

        Note: safetensors.safe_open doesn't expose a close() method. Releasing
        the mmap relies on reference counting: once file_handle_map is cleared
        and no tensor holds a reference to the underlying mmap region, the OS
        will reclaim the page cache. gc.collect() is called here to trigger
        immediate reclamation rather than waiting for the next GC cycle.
        """
        import gc
        self.file_handle_map.clear()
        gc.collect()

    def load_experts(self, base_key: str, device: str = "cpu"):
        """
        Load expert weights from SafeTensor files.

        Expected format:
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.weight
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.scale

        Args:
            base_key: Base key like "blk.{layer_index}"
            device: Target device for tensors

        Returns:
            Dictionary with keys: up, gate, down, up_scale, gate_scale, down_scale
            Each value is a list of lists: [numa_id][expert_id] -> numpy array
        """
        up_base_key = f"{base_key}.ffn_up_exps"
        gate_base_key = f"{base_key}.ffn_gate_exps"
        down_base_key = f"{base_key}.ffn_down_exps"
        max_numa_id = -1
        max_experts_count = -1
        while self.has_tensor(f"{up_base_key}.{max_experts_count+1}.numa.{0}.weight"):
            max_experts_count += 1
        # After the loop max_experts_count is the highest expert index found, i.e.
        # (expert count - 1). It stays -1 only when no experts exist for this key.
        # (Checking == 0 was an off-by-one: it falsely rejected single-expert layers
        # and silently accepted the zero-expert case, returning empty lists.)
        if max_experts_count == -1:
            raise ValueError(f"No experts found for key {base_key}")
        while self.has_tensor(f"{up_base_key}.{0}.numa.{max_numa_id+1}.weight"):
            max_numa_id += 1
        # Initialize empty lists to store tensors for each projection type
        up_weights = [[] for _ in range(max_numa_id + 1)]
        gate_weights = [[] for _ in range(max_numa_id + 1)]
        down_weights = [[] for _ in range(max_numa_id + 1)]
        up_scales = [[] for _ in range(max_numa_id + 1)]
        gate_scales = [[] for _ in range(max_numa_id + 1)]
        down_scales = [[] for _ in range(max_numa_id + 1)]
        # Check if backward weights exist
        up_bwd_base_key = f"{base_key}.ffn_up_bwd_exps"
        gate_bwd_base_key = f"{base_key}.ffn_gate_bwd_exps"
        down_bwd_base_key = f"{base_key}.ffn_down_bwd_exps"
        has_bwd = self.has_tensor(f"{gate_bwd_base_key}.{0}.numa.{0}.weight")

        if has_bwd:
            up_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            gate_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            down_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            up_bwd_scales = [[] for _ in range(max_numa_id + 1)]
            gate_bwd_scales = [[] for _ in range(max_numa_id + 1)]
            down_bwd_scales = [[] for _ in range(max_numa_id + 1)]

        for numa_id in range(max_numa_id + 1):
            for expert_id in range(max_experts_count + 1):
                up_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.weight"
                gate_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.weight"
                down_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.weight"
                up_scale_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.scale"
                gate_scale_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.scale"
                down_scale_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.scale"
                # make sure contiguous
                up_tensor = self.load_tensor(up_key, device).numpy()
                gate_tensor = self.load_tensor(gate_key, device).numpy()
                down_tensor = self.load_tensor(down_key, device).numpy()
                up_scale_tensor = self.load_tensor(up_scale_key, device).numpy()
                gate_scale_tensor = self.load_tensor(gate_scale_key, device).numpy()
                down_scale_tensor = self.load_tensor(down_scale_key, device).numpy()

                up_weights[numa_id].append(up_tensor)
                gate_weights[numa_id].append(gate_tensor)
                down_weights[numa_id].append(down_tensor)
                up_scales[numa_id].append(up_scale_tensor)
                gate_scales[numa_id].append(gate_scale_tensor)
                down_scales[numa_id].append(down_scale_tensor)

                # Load backward weights if available
                if has_bwd:
                    gate_bwd_weights[numa_id].append(
                        self.load_tensor(f"{gate_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    up_bwd_weights[numa_id].append(
                        self.load_tensor(f"{up_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    down_bwd_weights[numa_id].append(
                        self.load_tensor(f"{down_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    gate_bwd_scales[numa_id].append(
                        self.load_tensor(f"{gate_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )
                    up_bwd_scales[numa_id].append(
                        self.load_tensor(f"{up_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )
                    down_bwd_scales[numa_id].append(
                        self.load_tensor(f"{down_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )

        result = {
            "up": up_weights,
            "gate": gate_weights,
            "down": down_weights,
            "up_scale": up_scales,
            "gate_scale": gate_scales,
            "down_scale": down_scales,
        }
        if has_bwd:
            result["gate_bwd"] = gate_bwd_weights
            result["up_bwd"] = up_bwd_weights
            result["down_bwd"] = down_bwd_weights
            result["gate_bwd_scale"] = gate_bwd_scales
            result["up_bwd_scale"] = up_bwd_scales
            result["down_bwd_scale"] = down_bwd_scales
        return result

    def has_tensor(self, name: str):
        return name in self.tensor_file_map


class FP8SafeTensorLoader(SafeTensorLoader):
    """Loader for FP8 expert weights with auto-detection of naming formats.

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight
    - Mistral style: {base}.experts.{id}.{w1,w3,w2}.weight

    Supported scale formats (auto-detected):
    - Block-wise: weight_scale_inv (DeepSeek FP8)
    - Per-channel: weight_scale (GLM-4.7-FP8)

    The format is auto-detected during initialization.
    """

    # Known MoE naming formats: (experts_path_template, gate_name, up_name, down_name)
    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
        "mistral": ("{base}.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str, scale_suffix: str = None):
        """Initialize FP8 loader with optional scale suffix override.

        Args:
            file_path: Path to safetensor files
            scale_suffix: Optional scale key suffix. If None, auto-detect between
                         'weight_scale_inv' (block-wise) and 'weight_scale' (per-channel).
        """
        super().__init__(file_path)
        self._detected_format = None
        self._scale_suffix = scale_suffix  # None means auto-detect
        # Set per_channel based on explicit scale_suffix if provided
        if scale_suffix == "weight_scale":
            self._is_per_channel = True
        elif scale_suffix == "weight_scale_inv":
            self._is_per_channel = False
        else:
            self._is_per_channel = False  # Will be updated in _detect_format if auto-detect
        self._is_vl_model = False
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format and scale format by checking tensor keys."""
        # Sample some tensor names to detect format
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            # Check if any key matches this format pattern
            # Look for pattern like: model.layers.0.{experts_path}.0.{gate_name}.weight
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    # Verify the path template matches
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
                    elif fmt_name == "mistral" and ".mlp.experts" not in key and ".block_sparse_moe.experts" not in key:
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
            if self._detected_format:
                break

        # Default to deepseek if no format detected
        if not self._detected_format:
            self._detected_format = "deepseek"
            print("[FP8SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

        # Auto-detect scale suffix if not specified
        if self._scale_suffix is None:
            _, gate, _, _ = self.MOE_FORMATS[self._detected_format]
            # Check for per-channel scale (weight_scale) vs block-wise (weight_scale_inv)
            for key in sample_keys:
                if f".{gate}.weight_scale_inv" in key:
                    self._scale_suffix = "weight_scale_inv"
                    self._is_per_channel = False
                    print("[FP8SafeTensorLoader] Detected scale format: block-wise (weight_scale_inv)")
                    if key.startswith("model.language_model.") and self._detected_format == "deepseek":
                        # VL models(Qwen3.5): model.layers.{N} -> model.language_model.layers.{N}
                        self._is_vl_model = True
                        print("[FP8SafeTensorLoader] Detected VL model")
                    return
                elif f".{gate}.weight_scale" in key and "weight_scale_inv" not in key:
                    self._scale_suffix = "weight_scale"
                    # Some models (e.g., Mistral) use block-wise FP8 scales but keep
                    # the key suffix as `weight_scale` (without `_inv`). Infer format
                    # from scale tensor shape instead of suffix alone:
                    # - per-channel: [N] or [N, 1]
                    # - block-wise: [N_block, K_block] (both dims > 1)
                    scale_tensor = self.load_tensor(key, device="cpu")
                    if scale_tensor.dim() == 1:
                        self._is_per_channel = True
                    elif scale_tensor.dim() == 2 and scale_tensor.shape[1] == 1:
                        self._is_per_channel = True
                    else:
                        self._is_per_channel = False

                    scale_kind = "per-channel" if self._is_per_channel else "block-wise"
                    print(f"[FP8SafeTensorLoader] Detected scale format: {scale_kind} (weight_scale)")
                    return
            # Default to weight_scale_inv
            self._scale_suffix = "weight_scale_inv"
            self._is_per_channel = False
            print("[FP8SafeTensorLoader] No scale format detected, defaulting to: weight_scale_inv")
        else:
            # Scale suffix was explicitly provided
            scale_type = "per-channel" if self._is_per_channel else "block-wise"
            print(f"[FP8SafeTensorLoader] Using explicit scale format: {scale_type} ({self._scale_suffix})")

    def _get_experts_prefix_candidates(self, base_key: str) -> list[str]:
        """Get candidate experts prefixes based on detected format and base key variants."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        candidates = []
        if self._is_vl_model:
            base_key = base_key.replace("model.layers", "model.language_model.layers")
        candidates.append(path_tpl.format(base=base_key))

        # Some model weights (e.g., Mistral native format) do not have "model." prefix.
        if base_key.startswith("model."):
            candidates.append(path_tpl.format(base=base_key[len("model.") :]))

        # Deduplicate while preserving order.
        return list(dict.fromkeys(candidates))

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load FP8 expert weights and their scale tensors.

        Supports both block-wise (weight_scale_inv) and per-channel (weight_scale) formats.
        Per-channel scales are squeezed from [N, 1] to [N] if needed.
        """
        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"
            gate_s_key = f"{experts_prefix}.{exp_id}.{gate_name}.{self._scale_suffix}"
            up_s_key = f"{experts_prefix}.{exp_id}.{up_name}.{self._scale_suffix}"
            down_s_key = f"{experts_prefix}.{exp_id}.{down_name}.{self._scale_suffix}"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

            gate_scale = self.load_tensor(gate_s_key, device)
            up_scale = self.load_tensor(up_s_key, device)
            down_scale = self.load_tensor(down_s_key, device)

            # For per-channel scales, squeeze [N, 1] -> [N] if needed
            if self._is_per_channel:
                if gate_scale.dim() == 2 and gate_scale.shape[1] == 1:
                    gate_scale = gate_scale.squeeze(1)
                if up_scale.dim() == 2 and up_scale.shape[1] == 1:
                    up_scale = up_scale.squeeze(1)
                if down_scale.dim() == 2 and down_scale.shape[1] == 1:
                    down_scale = down_scale.squeeze(1)

            gate_scales[exp_id] = gate_scale.contiguous()
            up_scales[exp_id] = up_scale.contiguous()
            down_scales[exp_id] = down_scale.contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }

    def is_per_channel(self) -> bool:
        """Return True if using per-channel quantization, False for block-wise."""
        return self._is_per_channel


class BF16SafeTensorLoader(SafeTensorLoader):
    """Loader for native BF16 expert weights (no quantization, no scales).

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight
    - Mistral style: {base}.experts.{id}.{w1,w3,w2}.weight
    - Packed Qwen style: {base}.mlp.experts.{gate_up,down}_proj
    - Packed Gemma 4 style: {base}.experts.{gate_up,down}_proj

    The format is auto-detected during initialization.
    """

    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
        "mistral": ("{base}.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detected_format = None
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format by checking tensor keys."""
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        # Check for packed format first (Qwen3.5 MoE style: all experts in one 3D tensor)
        for key in sample_keys:
            if key.endswith((".mlp.experts.gate_up_proj", ".experts.gate_up_proj")):
                self._detected_format = "packed"
                print("[BF16SafeTensorLoader] Detected format: packed")
                return

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif fmt_name == "mistral" and ".mlp.experts" not in key and ".block_sparse_moe.experts" not in key:
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return

        self._detected_format = "deepseek"
        print("[BF16SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

    def _get_experts_prefix_candidates(self, base_key: str) -> list[str]:
        """Get candidate experts prefixes based on detected format and base key variants."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        candidates = [path_tpl.format(base=base_key)]

        # Some model weights (e.g., Mistral native format) do not have "model." prefix.
        if base_key.startswith("model."):
            candidates.append(path_tpl.format(base=base_key[len("model.") :]))

        return list(dict.fromkeys(candidates))

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load BF16 expert weights (no scales needed)."""
        # A mixed checkpoint can keep quantized target experts first and one
        # packed BF16 MTP layer near the end of the index. Global detection
        # from a key sample then describes the target, not this base_key.
        # Resolve the requested layer itself before using the global fallback.
        try:
            self._resolve_packed_experts_prefix(base_key)
        except ValueError:
            if self._detected_format == "packed":
                raise
        else:
            return self._load_experts_packed(base_key, device)

        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
        }

    def _resolve_packed_experts_prefix(self, base_key: str) -> str:
        """Resolve the experts prefix for packed format, trying fallbacks."""
        base_candidates = [base_key]

        # Multimodal checkpoints commonly insert language_model after model:
        # model.layers.{N} -> model.language_model.layers.{N}.
        if base_key.startswith("model.") and not base_key.startswith("model.language_model."):
            base_candidates.append(f"model.language_model.{base_key[len('model.'):]}")

        # Qwen stores the packed tensors below mlp.experts, while Gemma 4
        # stores them directly below layers.N.experts.
        prefix_candidates = []
        for candidate in base_candidates:
            prefix_candidates.extend(
                (f"{candidate}.mlp.experts", f"{candidate}.experts")
            )

        for experts_prefix in dict.fromkeys(prefix_candidates):
            if self.has_tensor(f"{experts_prefix}.gate_up_proj"):
                return experts_prefix

        raise ValueError(
            f"No packed experts found for base_key '{base_key}'. "
            f"Tried: {list(dict.fromkeys(prefix_candidates))}"
        )

    def _load_experts_packed(self, base_key: str, device: str = "cpu"):
        """Load packed expert weights (Qwen3.5/Gemma 4 style).

        Packed format stores all experts in stacked 3D tensors:
        - gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
        - down_proj:    [num_experts, hidden_size, intermediate_size]
        """
        experts_prefix = self._resolve_packed_experts_prefix(base_key)

        gate_up_key = f"{experts_prefix}.gate_up_proj"
        down_key = f"{experts_prefix}.down_proj"

        gate_up = self.load_tensor(gate_up_key, device)  # [E, 2*I, H]
        down = self.load_tensor(down_key, device)  # [E, H, I]

        mid = gate_up.shape[1] // 2
        gate_list = [gate_up[i, :mid, :].contiguous() for i in range(gate_up.shape[0])]
        up_list = [gate_up[i, mid:, :].contiguous() for i in range(gate_up.shape[0])]
        down_list = [down[i].contiguous() for i in range(down.shape[0])]

        return {
            "gate": gate_list,
            "up": up_list,
            "down": down_list,
        }


class CompressedSafeTensorLoader(SafeTensorLoader):
    """Loader for compressed SafeTensor layouts (RAWINT4 weights)."""

    @staticmethod
    def _normalize_rawint4_weight(weight_tensor, scale_tensor, shape_tensor=None, key: str = "weight_packed"):
        """Return byte-packed uint8 RAWINT4 weights expected by kt_kernel_ext."""
        if weight_tensor.dtype == torch.int32:
            # compressed-tensors pack-quantized stores 8 int4 values per int32.
            # The RAWINT4 kernels consume the same bytes as uint8, two int4 values per byte.
            rows, int32_cols = weight_tensor.shape
            weight_tensor = weight_tensor.contiguous().view(torch.uint8).view(rows, int32_cols * 4).contiguous()
        elif weight_tensor.dtype == torch.uint8:
            weight_tensor = weight_tensor.contiguous()
        else:
            raise TypeError(f"{key} must be torch.uint8 or torch.int32, got {weight_tensor.dtype}")

        if shape_tensor is None:
            return weight_tensor

        shape_values = shape_tensor.detach().cpu().tolist()
        if len(shape_values) != 2:
            raise ValueError(f"{key}.weight_shape must contain [out_features, in_features], got {shape_values}")

        out_features, in_features = (int(shape_values[0]), int(shape_values[1]))
        if out_features <= 0 or in_features <= 0:
            return weight_tensor

        if in_features % 2 != 0:
            return weight_tensor

        expected_weight_shape = (out_features, in_features // 2)
        if tuple(weight_tensor.shape) != expected_weight_shape:
            return weight_tensor

        if scale_tensor.dim() != 2 or scale_tensor.shape[0] != out_features or scale_tensor.shape[1] <= 0:
            raise ValueError(
                f"{key} scale shape {tuple(scale_tensor.shape)} is incompatible with weight_shape={shape_values}"
            )

        if in_features % int(scale_tensor.shape[1]) != 0:
            raise ValueError(
                f"{key} in_features={in_features} is not divisible by scale columns={scale_tensor.shape[1]}"
            )

        return weight_tensor

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load raw expert weights stored in compressed safetensor format."""

        experts_prefix = f"{base_key}.mlp.experts"

        expert_idx = 0
        while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
            expert_idx += 1

        if expert_idx == 0:
            experts_prefix = f"language_model.{base_key}.mlp.experts"
            expert_idx = 0
            while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
                expert_idx += 1
            if expert_idx == 0:
                raise ValueError(f"No experts found for key {experts_prefix}")

        def load_projection(proj_name: str):
            weight_entries = []
            scale_entries = []

            for exp_id in range(expert_idx):
                weight_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_packed"
                scale_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_scale"
                shape_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_shape"

                if not self.has_tensor(weight_key):
                    raise KeyError(f"Missing tensor: {weight_key}")
                if not self.has_tensor(scale_key):
                    raise KeyError(f"Missing tensor: {scale_key}")

                weight_tensor = self.load_tensor(weight_key, device).contiguous()
                scale_tensor = self.load_tensor(scale_key, device).contiguous()
                shape_tensor = self.load_tensor(shape_key, "cpu") if self.has_tensor(shape_key) else None
                weight_tensor = self._normalize_rawint4_weight(weight_tensor, scale_tensor, shape_tensor, weight_key)

                weight_entries.append(weight_tensor)
                scale_entries.append(scale_tensor)

            return weight_entries, scale_entries

        gate_weights, gate_scales = load_projection("gate")
        up_weights, up_scales = load_projection("up")
        down_weights, down_scales = load_projection("down")

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class GGUFLoader:
    """
    GGUF format loader using the official gguf library (gguf.gguf_reader.GGUFReader)

    This is a cleaner implementation compared to manual binary parsing.
    """

    def __init__(self, gguf_path: str):
        """
        Initialize GGUF loader from a file or directory

        Args:
            gguf_path: Path to a single GGUF file or a directory containing GGUF files
        """
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF path not found: {gguf_path}")

        self.tensor_info = {}
        self.metadata = {}
        self.tensor_file_map = {}
        self.file_data_map = {}

        if os.path.isfile(gguf_path) and gguf_path.endswith(".gguf"):
            print(f"\n[GGUFLoader] Loading single GGUF file : {os.path.basename(gguf_path)}")
            self._load_single_file(gguf_path)
        elif os.path.isdir(gguf_path):
            print(f"\n[GGUFLoader] Loading GGUF files from directory: {gguf_path}")
            self._load_directory(gguf_path)
        else:
            raise ValueError(f"Path must be a .gguf file or a directory: {gguf_path}")

        print(f"[GGUFLoader] Summary:")
        print(f"  Files loaded: {len(self.file_data_map)}")
        print(f"  Total tensors: {len(self.tensor_info)}")
        print(f"  Metadata keys: {len(self.metadata)}")
        tensors = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]
        for key in tensors:
            if key in self.tensor_info:
                info = self.tensor_info[key]
                print(f" {'.'.join(key.split('.')[2:-1])}, Dtype: {info['dtype'].name}")

    def _load_single_file(self, file_path: str):
        """Load a single GGUF file"""
        reader = GGUFReader(file_path)

        for key, field in reader.fields.items():
            value = field.parts[field.data[0]]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                try:
                    value = bytes(value).decode("utf-8")
                except:
                    pass
            self.metadata[key] = value

        for tensor in reader.tensors:
            self.tensor_info[tensor.name] = {
                "shape": list(reversed(tensor.shape)),  # Reverse to match PyTorch order
                "dtype": tensor.tensor_type,
                "offset": tensor.data_offset,
                "n_elements": tensor.n_elements,
            }
            self.tensor_file_map[tensor.name] = file_path

        self.file_data_map[file_path] = np.memmap(file_path, mode="r")

    def _load_directory(self, dir_path: str):
        """Load all GGUF files from a directory (non-recursive)"""
        found_gguf = False

        for file in sorted(os.listdir(dir_path)):
            if file.endswith(".gguf"):
                found_gguf = True
                file_path = os.path.join(dir_path, file)
                print(f"  Loading: {file}")

                reader = GGUFReader(file_path)

                for key, field in reader.fields.items():
                    value = field.parts[field.data[0]]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                        try:
                            value = bytes(value).decode("utf-8")
                        except:
                            pass
                    self.metadata[key] = value

                for tensor in reader.tensors:
                    self.tensor_info[tensor.name] = {
                        "shape": list(reversed(tensor.shape)),
                        "dtype": tensor.tensor_type,
                        "offset": tensor.data_offset,
                        "n_elements": tensor.n_elements,
                    }
                    self.tensor_file_map[tensor.name] = file_path

                self.file_data_map[file_path] = np.memmap(file_path, mode="r")

        if not found_gguf:
            raise FileNotFoundError(f"No .gguf files found in directory: {dir_path}")

    def get_model_config(self, layer_idx: int = 0):
        """
        Extract model configuration from GGUF metadata and tensor shapes.

        Args:
            layer_idx: Layer index to inspect (default: 0)

        Returns:
            dict with keys: num_experts, num_experts_per_tok, hidden_size, moe_intermediate_size
        """
        config = {}

        arch = self.metadata.get("general.architecture", "unknown")

        num_experts = None
        for key_suffix in [
            "expert_count",
            "expert.count",
            "moe.expert_count",
            "expert_feed_forward_length",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        num_experts_per_tok = None
        for key_suffix in [
            "expert_used_count",
            "expert.used_count",
            "moe.num_experts_per_tok",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts_per_tok = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        hidden_size = None
        for key_suffix in [
            "embedding_length",
            "embed_length",
            "hidden_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                hidden_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        moe_intermediate_size = None
        for key_suffix in [
            "expert_feed_forward_length",
            "feed_forward_length",
            "ffn_length",
            "intermediate_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                moe_intermediate_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        if any(v is None for v in [num_experts, hidden_size, moe_intermediate_size]):

            base_key = f"blk.{layer_idx}.ffn_gate_exps.weight"
            if base_key in self.tensor_info:
                gate_shape = self.tensor_info[base_key]["shape"]
                print(f"  Found tensor '{base_key}' with shape: {gate_shape}")

                if len(gate_shape) >= 3:
                    if num_experts is None:
                        num_experts = int(gate_shape[0])
                    if moe_intermediate_size is None:
                        moe_intermediate_size = int(gate_shape[1])
                    if hidden_size is None:
                        hidden_size = int(gate_shape[2])

        config = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
        }

        return config

    def print_metadata(self, filter_keywords=None):
        """
        Print GGUF file metadata for debugging.

        Args:
            filter_keywords: Optional list of keywords to filter metadata keys
        """
        print(f"\n[GGUFLoader] GGUF Metadata:")
        print(f"  Total metadata entries: {len(self.metadata)}")

        if filter_keywords:
            filtered = {
                k: v for k, v in self.metadata.items() if any(kw.lower() in k.lower() for kw in filter_keywords)
            }
            for k, v in sorted(filtered.items()):
                print(f"  {k}: {v}")
        else:
            for k, v in sorted(self.metadata.items()):
                print(f"  {k}: {v}")

    def has_tensor(self, name: str):
        """Check if tensor exists"""
        name = translate_name_to_gguf(name)
        return name in self.tensor_info

    def get_ggml_type(self, name: str):
        """Get GGML type of a tensor"""
        name = translate_name_to_gguf(name)
        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")
        return self.tensor_info[name]["dtype"]

    def get_undequanted_tensor_and_ggml_type(self, name: str):
        """
        Get tensor data and its GGML type without dequantizing

        Args:
            name: Tensor name (in PyTorch format, will be translated to GGUF format)

        Returns:
            (data, ggml_type): Tuple of tensor data and GGML quantization type
        """
        name = translate_name_to_gguf(name)

        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")

        info = self.tensor_info[name]
        file_path = self.tensor_file_map[name]
        mmap_data = self.file_data_map[file_path]

        offset = info["offset"]
        n_elements = info["n_elements"]
        ggml_type = info["dtype"]

        GGML_QUANT_SIZES = {
            GGMLQuantizationType.F32: (1, 4),
            GGMLQuantizationType.F16: (1, 2),
            GGMLQuantizationType.BF16: (1, 2),
            GGMLQuantizationType.Q4_0: (32, 2 + 16),
            GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
            GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
            GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
            GGMLQuantizationType.Q8_0: (32, 2 + 32),
            GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
            GGMLQuantizationType.Q2_K: (256, 2 + 2 + 256 // 16 + 256 // 4),
            GGMLQuantizationType.Q3_K: (256, 2 + 256 // 4 + 256 // 8 + 12),
            GGMLQuantizationType.Q4_K: (256, 2 + 2 + 256 // 2 + 12),
            GGMLQuantizationType.Q5_K: (256, 2 + 2 + 256 // 2 + 256 // 8 + 12),
            GGMLQuantizationType.Q6_K: (256, 2 + 256 // 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.Q8_K: (256, 4 + 256 + 256 // 8),
            GGMLQuantizationType.IQ2_XXS: (256, 2 + 256 // 4),
            GGMLQuantizationType.IQ2_XS: (256, 2 + 256 // 4 + 256 // 32),
            GGMLQuantizationType.IQ3_XXS: (256, 2 + 256 // 4 + 256 // 8),
            GGMLQuantizationType.IQ1_S: (256, 2 + 256 // 8 + 256 // 16),
            GGMLQuantizationType.IQ4_NL: (32, 2 + 16),
            GGMLQuantizationType.IQ3_S: (256, 2 + 256 // 4 + 256 // 8 + 256 // 32 + 4),
            GGMLQuantizationType.IQ2_S: (256, 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + 256 // 2 + 256 // 64),
            GGMLQuantizationType.I8: (1, 1),
            GGMLQuantizationType.I16: (1, 2),
            GGMLQuantizationType.I32: (1, 4),
            GGMLQuantizationType.I64: (1, 8),
            GGMLQuantizationType.F64: (1, 8),
            GGMLQuantizationType.IQ1_M: (256, 256 // 8 + 256 // 16 + 256 // 32),
        }
        for extra_name, extra_sizes in _EXTRA_GGML_QUANT_SIZES.items():
            extra = getattr(GGMLQuantizationType, extra_name, None)
            if extra is not None:
                GGML_QUANT_SIZES[extra] = extra_sizes
        if ggml_type not in GGML_QUANT_SIZES and getattr(ggml_type, "value", None) in _EXTRA_GGML_TYPES_BY_ID:
            GGML_QUANT_SIZES[ggml_type] = _EXTRA_GGML_TYPES_BY_ID[ggml_type.value][1]

        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        n_bytes = n_elements * type_size // block_size

        data_bytes = mmap_data[offset : offset + n_bytes]
        data = torch.from_numpy(np.frombuffer(data_bytes, dtype=np.uint8).copy())

        return data, ggml_type


class GPTQSafeTensorLoader(FP8SafeTensorLoader):
    """Loader for symmetric GPTQ-Int4 expert weights (qweight + scales, no qzeros).

    Only supports sym=true, desc_act=false GPTQ models.

    Tensor keys:
    - qweight: {prefix}.{id}.{proj}.qweight  (int32, packed 8x4-bit along K)
    - scales:  {prefix}.{id}.{proj}.scales    (fp16 -> converted to fp32)
    """

    def __init__(self, file_path: str):
        # Call FP8SafeTensorLoader init (which calls SafeTensorLoader init + format detection)
        super().__init__(file_path, scale_suffix="scales")
        # Verify GPTQ config
        self._verify_gptq_config(file_path)

    def _detect_format(self):
        """Override FP8 format detection to look for .qweight instead of .weight."""
        sample_keys = list(self.tensor_file_map.keys())[:2000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.qweight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        break
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        # Check for VL model (language_model prefix)
                        if "language_model." in key:
                            self._is_vl_model = True
                        break
                    elif fmt_name == "mistral" and "block_sparse_moe" not in key and "mlp" not in key:
                        self._detected_format = fmt_name
                        break
            if self._detected_format is not None:
                break

        if self._detected_format is None:
            self._detected_format = "deepseek"

        vl_str = " (VL model)" if self._is_vl_model else ""
        print(f"[GPTQSafeTensorLoader] Detected format: {self._detected_format}{vl_str}")

    def _verify_gptq_config(self, file_path):
        """Check that the model uses sym=true, desc_act=false."""
        import json
        import os

        config_path = os.path.join(os.path.dirname(file_path), "config.json")
        if not os.path.exists(config_path):
            # Try parent directory
            config_path = os.path.join(file_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            qc = config.get("quantization_config", {})
            if qc.get("quant_method") == "gptq":
                if qc.get("desc_act", False):
                    raise NotImplementedError(
                        "GPTQ desc_act=true is not supported. Only desc_act=false models are supported."
                    )
                if not qc.get("sym", True):
                    raise NotImplementedError(
                        "GPTQ sym=false (asymmetric) is not supported. Only sym=true models are supported."
                    )
                print(f"[GPTQSafeTensorLoader] Verified: sym={qc.get('sym')}, desc_act={qc.get('desc_act')}, "
                      f"bits={qc.get('bits')}, group_size={qc.get('group_size')}")

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load GPTQ expert qweight and scales.

        Returns dict with keys: gate, up, down (qweight int32), gate_scale, up_scale, down_scale (fp32).
        """
        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.qweight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No GPTQ experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            gate_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{gate_name}.qweight", device).contiguous()
            up_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{up_name}.qweight", device).contiguous()
            down_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{down_name}.qweight", device).contiguous()

            gate_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{gate_name}.scales", device).float().contiguous()
            up_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{up_name}.scales", device).float().contiguous()
            down_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{down_name}.scales", device).float().contiguous()

        print(f"[GPTQSafeTensorLoader] Loaded {expert_count} experts from {experts_prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class MXFP4SafeTensorLoader(SafeTensorLoader):
    """Loader for native MXFP4 expert weights (DeepSeek-V4-Flash format).

    Per expert layout:
      {base}.ffn.experts.{i}.w1.weight  I8       [N, K/2]   nibble-packed E2M1 (gate)
      {base}.ffn.experts.{i}.w1.scale   F8_E8M0  [N, K/32]  ue8m0 group scale
      {base}.ffn.experts.{i}.w3.{weight,scale}              up
      {base}.ffn.experts.{i}.w2.{weight,scale}              down

    V4 ckpt keys are not prefixed with ``model.``; we also probe the stripped form so
    callers can keep passing ``base_key="model.layers.{L}"``. ue8m0 → bf16 is a lossless
    bit shift (both have an 8-bit exponent and zero mantissa for ue8m0), and the AMX
    FP4 backend already consumes bf16 scales.
    """

    EXPERTS_PATH_TPL = "{base}.ffn.experts"
    PROJ_NAMES = ("w1", "w3", "w2")  # (gate, up, down)

    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        if base_key.startswith("model."):
            candidates.append(self.EXPERTS_PATH_TPL.format(base=base_key[len("model.") :]))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _ue8m0_to_bf16(scale_t: torch.Tensor) -> torch.Tensor:
        if scale_t.dtype != torch.uint8:
            scale_t = scale_t.view(torch.uint8)
        # bf16 = [sign(1) | exp(8) | mant(7)]; setting mant=0, exp=e gives 2^(e-127),
        # which is exactly the value encoded by ue8m0 for e ∈ [1, 254]. e=0 → bf16 +0
        # (acceptable: ue8m0=0 represents 2^-127, below bf16 normal range), e=255 → +inf.
        # Compute in int32 then narrow to int16 (max value is 255<<7=32640, fits int16),
        # because torch CPU has no lshift kernel for uint16.
        return (scale_t.to(torch.int32) << 7).to(torch.int16).view(torch.bfloat16).contiguous()

    def load_experts(self, base_key: str, device: str = "cpu"):
        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{cand}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                prefix = cand
                break
        if prefix is None:
            raise ValueError(
                f"No MXFP4 experts found under any of: {self._experts_prefix_candidates(base_key)}"
            )

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            for proj, dst in (
                (gate_name, gate_weights),
                (up_name, up_weights),
                (down_name, down_weights),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight", device).contiguous()
                if w.dtype != torch.uint8:
                    w = w.view(torch.uint8)
                dst[exp_id] = w

            for proj, dst in (
                (gate_name, gate_scales),
                (up_name, up_scales),
                (down_name, down_scales),
            ):
                s = self.load_tensor(f"{prefix}.{exp_id}.{proj}.scale", device)
                dst[exp_id] = self._ue8m0_to_bf16(s)

        print(f"[MXFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class NVFP4SafeTensorLoader(SafeTensorLoader):
    """Loader for ModelOpt NVFP4 expert weights (W4A16_NVFP4, group_size 16).

    Per expert layout (compressed-tensors / ModelOpt naming):
      {base}.mlp.experts.{i}.gate_proj.weight          U8        [N, K/2]   nibble-packed E2M1
      {base}.mlp.experts.{i}.gate_proj.weight_scale    F8_E4M3   [N, K/16]  per-block scale
      {base}.mlp.experts.{i}.gate_proj.weight_scale_2  F32       scalar     per-tensor global
      {base}.mlp.experts.{i}.{up_proj,down_proj}.{weight,weight_scale,weight_scale_2}

    NVFP4 shares MXFP4's E2M1 codepoints and low-nibble-first packing, so the
    weight bytes go to the FP4 kernel untouched; only the scale encoding differs.
    The kernel consumes one FP32-valued scale per k-group, so both levels are
    folded here into a single bf16 tensor:

        scale[n, g] = float(e4m3(weight_scale[n, g])) * weight_scale_2

    bf16 is lossy for E4M3 products (8 mantissa bits down to 7), which is what
    the kernel's own group-32 MXFP4 path already accepts; measured against a
    float64 dequant of the same bytes the end-to-end error is bf16-rounding
    only (cosine 0.999993 on Ornith-1.5-35B-A3B-NVFP4).
    """

    EXPERTS_PATH_TPL = "{base}.mlp.experts"
    PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        for strip in ("model.language_model.", "language_model.model.", "language_model.", "model."):
            if base_key.startswith(strip):
                candidates.append(self.EXPERTS_PATH_TPL.format(base=base_key[len(strip):]))
        # ModelOpt checkpoints for multimodal Qwen3.5-MoE nest under
        # model.language_model.layers.{L}; callers may pass either form.
        if not base_key.startswith("model.language_model."):
            for pre in ("model.language_model.", "language_model.model."):
                if base_key.startswith("model."):
                    candidates.append(self.EXPERTS_PATH_TPL.format(base=pre + base_key[len("model."):]))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _fold_scales_to_bf16(block_scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
        """e4m3 block scale x per-tensor f32 global -> bf16 [N, K/16]."""
        s = block_scale.to(torch.float32) * global_scale.to(torch.float32).reshape(())
        return s.to(torch.bfloat16).contiguous()

    def load_experts(self, base_key: str, device: str = "cpu"):
        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{cand}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                prefix = cand
                break
        if prefix is None:
            raise ValueError(
                f"No NVFP4 experts found under any of: {self._experts_prefix_candidates(base_key)}"
            )

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            for proj, wdst, sdst in (
                (gate_name, gate_weights, gate_scales),
                (up_name, up_weights, up_scales),
                (down_name, down_weights, down_scales),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight", device).contiguous()
                if w.dtype != torch.uint8:
                    w = w.view(torch.uint8)
                wdst[exp_id] = w

                bs = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale", device)
                gs = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale_2", device)
                sdst[exp_id] = self._fold_scales_to_bf16(bs, gs)

        print(f"[NVFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class MXFP8SafeTensorLoader(SafeTensorLoader):
    """Loader for native MXFP8 expert weights (MiniMax M3 Preview format).

    Per expert layout:
      {base}.block_sparse_moe.experts.{i}.w1.weight            F8_E4M3  [N, K]     gate
      {base}.block_sparse_moe.experts.{i}.w1.weight_scale_inv  U8       [N, K/32]  ue8m0
      {base}.block_sparse_moe.experts.{i}.w3.{weight,weight_scale_inv}             up
      {base}.block_sparse_moe.experts.{i}.w2.{weight,weight_scale_inv}             down

    M3 keys are prefixed with ``language_model.model.layers.{L}``; we also probe
    the stripped form. Scales stay as uint8 — the C++ kernel converts ue8m0→FP32
    via bit-shift during load_weights.
    """

    EXPERTS_PATH_TPL = "{base}.block_sparse_moe.experts"
    PROJ_NAMES = ("w1", "w3", "w2")  # (gate, up, down)

    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        for strip in ("language_model.model.", "language_model.", "model."):
            if base_key.startswith(strip):
                candidates.append(self.EXPERTS_PATH_TPL.format(base=base_key[len(strip):]))
        return list(dict.fromkeys(candidates))

    def load_experts(self, base_key: str, device: str = "cpu"):
        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{cand}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                prefix = cand
                break
        if prefix is None:
            raise ValueError(
                f"No MXFP8 experts found under any of: {self._experts_prefix_candidates(base_key)}"
            )

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            for proj, dst in (
                (gate_name, gate_weights),
                (up_name, up_weights),
                (down_name, down_weights),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight", device).contiguous()
                if w.dtype != torch.uint8:
                    w = w.view(torch.uint8)
                dst[exp_id] = w

            for proj, dst in (
                (gate_name, gate_scales),
                (up_name, up_scales),
                (down_name, down_scales),
            ):
                s = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale_inv", device).contiguous()
                if s.dtype != torch.uint8:
                    s = s.view(torch.uint8)
                dst[exp_id] = s

        print(f"[MXFP8SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


# --- KT_GGUF_FAST_HEADER (patches/kt-kernel-gguf-fast-header.py) ---
import struct as _kt_struct

_KT_FAST_HEADER = os.environ.get("KT_GGUF_FAST_HEADER") == "1"
_kt_header_cache = {}
_KT_SCALAR = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
              6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}
_KT_STRING, _KT_ARRAY = 8, 9


def _kt_meta_value(value):
    """GGUFLoader's conversion of parts[data[0]] (kept identical)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
        try:
            value = bytes(value).decode("utf-8")
        except:  # noqa: E722  (same as the loader)
            pass
    return value


def _kt_parse_gguf_header(file_path):
    """(metadata, tensor_info, memmap) as GGUFLoader builds them from GGUFReader."""
    mm = np.memmap(file_path, mode="r")
    u32, u64 = _kt_struct.Struct("<I"), _kt_struct.Struct("<Q")
    if u32.unpack_from(mm, 0)[0] != _gguf_constants.GGUF_MAGIC:
        raise ValueError("GGUF magic invalid")
    version = u32.unpack_from(mm, 4)[0]
    if version not in _gguf_reader.READER_SUPPORTED_VERSIONS:
        raise ValueError(f"Sorry, file appears to be version {version} which we cannot handle")
    n_tensors, n_kv = _kt_struct.unpack_from("<QQ", mm, 8)

    def arr(off, dtype, count=1):
        return np.frombuffer(mm, dtype=dtype, count=count, offset=off)

    raw = {"GGUF.version": arr(4, np.uint32), "GGUF.tensor_count": arr(8, np.uint64),
           "GGUF.kv_count": arr(16, np.uint64)}
    types = {}

    def value_at(off, vt):
        """(first value as GGUFReader's parts[data[0]], offset after the value)."""
        if vt == _KT_STRING:
            n = u64.unpack_from(mm, off)[0]
            return arr(off + 8, np.uint8, n), off + 8 + n
        dt = _KT_SCALAR.get(vt)
        if dt is not None:
            return arr(off, dt), off + np.dtype(dt).itemsize
        if vt == _KT_ARRAY:
            it = u32.unpack_from(mm, off)[0]
            n = u64.unpack_from(mm, off + 4)[0]
            off += 12
            first = None
            if it == _KT_STRING:
                for i in range(n):
                    ln = u64.unpack_from(mm, off)[0]
                    if i == 0:
                        first = arr(off + 8, np.uint8, ln)
                    off += 8 + ln
                return first, off
            dt = _KT_SCALAR.get(it)
            if dt is not None:
                if n:
                    first = arr(off, dt)
                return first, off + n * np.dtype(dt).itemsize
            # nested arrays: element by element
            for i in range(n):
                v, off = value_at(off, it)
                if i == 0:
                    first = v
            return first, off
        raise ValueError(f"Unknown/unhandled field type {vt}")

    off = 24
    for _ in range(n_kv):
        kl = u64.unpack_from(mm, off)[0]
        key = bytes(mm[off + 8:off + 8 + kl]).decode("utf-8")
        off += 8 + kl
        vt = u32.unpack_from(mm, off)[0]
        off += 4
        if key in raw:
            raise KeyError(f"Duplicate {key} already in list at offset {off}")
        v, off = value_at(off, vt)
        if v is None:
            raise IndexError(f"GGUF field {key} has no value (GGUFLoader reads data[0])")
        raw[key] = v
        types[key] = vt
    tinfo = []
    for _ in range(n_tensors):
        nl = u64.unpack_from(mm, off)[0]
        name = str(bytes(mm[off + 8:off + 8 + nl]), encoding="utf-8")
        off += 8 + nl
        nd = u32.unpack_from(mm, off)[0]
        off += 4
        dims = arr(off, np.uint64, nd)
        off += 8 * nd
        rt = u32.unpack_from(mm, off)[0]
        toff = u64.unpack_from(mm, off + 4)[0]
        off += 12
        tinfo.append((name, dims, rt, toff))
    align = 32
    if "general.alignment" in raw:
        if types["general.alignment"] != 4:
            raise ValueError("Bad type for general.alignment field")
        align = int(raw["general.alignment"][0])
        if align == 0 or (align & (align - 1)) != 0:
            raise ValueError("Invalid alignment: must be a non-zero power of two")
    if off % align:
        off += align - off % align
    metadata = {k: _kt_meta_value(v) for k, v in raw.items()}
    tensors = {}
    for name, dims, rt, toff in tinfo:
        if name in tensors:
            raise ValueError(f"Found duplicated tensor with name {name}")
        tensors[name] = {
            "shape": list(reversed(dims)),
            "dtype": _gguf_reader.GGMLQuantizationType(rt),
            "offset": int(off + toff),
            "n_elements": int(np.prod(dims)),
        }
    return metadata, tensors, mm


def _kt_gguf_header(file_path):
    st = os.stat(file_path)
    key = (os.path.realpath(file_path), st.st_size, st.st_mtime_ns)
    hit = _kt_header_cache.get(key)
    if hit is None:
        hit = _kt_header_cache[key] = _kt_parse_gguf_header(file_path)
    return hit


def _kt_fast_single(self, file_path: str):
    metadata, tensors, mm = _kt_gguf_header(file_path)
    self.metadata.update(metadata)
    for name, info in tensors.items():
        self.tensor_info[name] = dict(info)
        self.tensor_file_map[name] = file_path
    self.file_data_map[file_path] = mm


def _kt_fast_directory(self, dir_path: str):
    found_gguf = False
    for file in sorted(os.listdir(dir_path)):
        if file.endswith(".gguf"):
            found_gguf = True
            file_path = os.path.join(dir_path, file)
            print(f"  Loading: {file}")
            _kt_fast_single(self, file_path)
    if not found_gguf:
        raise FileNotFoundError(f"No .gguf files found in directory: {dir_path}")


if _KT_FAST_HEADER:
    GGUFLoader._load_single_file = _kt_fast_single
    GGUFLoader._load_directory = _kt_fast_directory


# --- KT_GGUF_NOCOPY (patches/kt-kernel-gguf-nocopy.py) ---
def _kt_get_tensor_view(self, name: str):
    """(uint8 tensor over the file mapping, ggml type): the bytes
    get_undequanted_tensor_and_ggml_type would copy, not copied. Read-only."""
    import warnings

    data, qt = None, None
    tname = translate_name_to_gguf(name)
    if tname not in self.tensor_info:
        raise KeyError(f"Tensor '{tname}' not found in GGUF files")
    info = self.tensor_info[tname]
    mm = self.file_data_map[self.tensor_file_map[tname]]
    qt = info["dtype"]
    sizes = dict(_gguf_constants.GGML_QUANT_SIZES)
    for extra_name, extra_sizes in _EXTRA_GGML_QUANT_SIZES.items():
        extra = getattr(_gguf_constants.GGMLQuantizationType, extra_name, None)
        if extra is not None:
            sizes[extra] = extra_sizes
    if qt not in sizes and getattr(qt, "value", None) in _EXTRA_GGML_TYPES_BY_ID:
        sizes[qt] = _EXTRA_GGML_TYPES_BY_ID[qt.value][1]
    block_size, type_size = sizes[qt]
    n_bytes = info["n_elements"] * type_size // block_size
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_bytes, offset=info["offset"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # read-only buffer
        data = torch.from_numpy(raw)
    return data, qt


GGUFLoader.get_tensor_view = _kt_get_tensor_view


class EXL3SafeTensorLoader(SafeTensorLoader):
    """ExLlamaV3 EXL3 expert weights (quantization_config.quant_method == "exl3", codebook mul1).

    Per expert and projection: {base}.mlp.experts.{i}.{gate,up,down}_proj.trellis  I16 [K/16, N/16, 16*bits]
    (K inputs, N outputs, 256-weight tiles), .suh F16 [K] and .svh F16 [N] (the sign vectors of the
    blockwise Hadamard transforms), .mul1 I32 (the codebook multiplier, 0x83DCD12D). The tensors go to
    the kernel as they are; bits come from the trellis width."""

    EXPERTS_PATH_TPL = "{base}.mlp.experts"
    PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        for strip in ("model.language_model.", "language_model.model.", "language_model.", "model."):
            if base_key.startswith(strip):
                candidates.append(self.EXPERTS_PATH_TPL.format(base=base_key[len(strip):]))
        if not base_key.startswith("model.language_model."):
            for pre in ("model.language_model.", "language_model.model."):
                if base_key.startswith("model."):
                    candidates.append(self.EXPERTS_PATH_TPL.format(base=pre + base_key[len("model."):]))
        return list(dict.fromkeys(candidates))

    def load_experts(self, base_key: str, device: str = "cpu"):
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{cand}.{expert_count}.gate_proj.trellis"):
                expert_count += 1
            if expert_count > 0:
                prefix = cand
                break
        if prefix is None:
            raise ValueError(f"No EXL3 experts found under any of: {self._experts_prefix_candidates(base_key)}")
        out = {"gate": [None] * expert_count, "up": [None] * expert_count, "down": [None] * expert_count}
        for proj in ("gate", "up", "down"):
            out[f"{proj}_suh"] = [None] * expert_count
            out[f"{proj}_svh"] = [None] * expert_count
        bits = {}

        def load_one(i):
            for proj in ("gate", "up", "down"):
                key = f"{prefix}.{i}.{proj}_proj"
                t = self.load_tensor(f"{key}.trellis", device)
                if t.dtype != torch.int16 or t.dim() != 3:
                    raise ValueError(f"{key}.trellis: expected int16 [K/16, N/16, 16*bits], got {t.dtype} {tuple(t.shape)}")
                if not t.is_contiguous():
                    t = t.contiguous()
                suh = self.load_tensor(f"{key}.suh", device)
                svh = self.load_tensor(f"{key}.svh", device)
                if suh.dtype != torch.float16 or svh.dtype != torch.float16:
                    raise ValueError(f"{key}: suh / svh must be fp16")
                mul1 = int(self.load_tensor(f"{key}.mul1", device).item()) & 0xFFFFFFFF
                if mul1 != 0x83DCD12D:
                    raise ValueError(f"{key}.mul1 = {mul1:#x}: only the mul1 codebook (0x83dcd12d) is supported")
                w = t.shape[2]
                if w % 16 != 0:
                    raise ValueError(f"{key}.trellis: half-integer bit rates are not supported (tile width {w})")
                b = w // 16
                if bits.setdefault(proj, b) != b:
                    raise ValueError(f"{key}: bits differ between experts")
                out[proj][i] = t
                out[f"{proj}_suh"][i] = suh.contiguous()
                out[f"{proj}_svh"][i] = svh.contiguous()

        if hasattr(self, "_for_experts"):
            self._for_experts(expert_count, load_one)
        else:
            for i in range(expert_count):
                load_one(i)
        out["bits"] = bits
        return out
