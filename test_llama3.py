"""Test Llama 3.1 ONNX model with ONNX Runtime and text generation."""

import argparse
import pathlib
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_iree
from transformers import AutoTokenizer

MODEL_PATH = "llama3.1-onnx/model.onnx"
TOKENIZER_PATH = "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# IREE vendor ID for device allocation
IREE_VENDOR_ID = 0x1EEE


def print_model_info(session):
    """Print detailed model input/output information."""
    print("\n=== Model Inputs ===")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

    print("\n=== Model Outputs (first 5) ===")
    for out in session.get_outputs()[:5]:
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
    print(f"  ... ({len(session.get_outputs())} total outputs)")


def get_iree_session(model_path: str, target_arch: str = "llvm-cpu"):
    """Create an ONNX Runtime session with the IREE EP."""
    ort.set_default_logger_severity(0)

    ep_lib_path = onnxruntime_ep_iree.get_library_path()

    print(f"EP library path: {ep_lib_path}")
    ort.register_execution_provider_library("IREE", str(ep_lib_path))
    print("EP plugin registered successfully")

    # Determine driver from target_arch
    if "vulkan" in target_arch:
        iree_driver = "vulkan"
    elif "hip" in target_arch or "rocm" in target_arch:
        iree_driver = "hip"
    else:
        iree_driver = "local-task"

    ep_devices = ort.get_ep_devices()
    iree_device = None
    for dev in ep_devices:
        if dev.device.metadata.get("iree.driver") == iree_driver:
            iree_device = dev
            break

    if not iree_device:
        available = [d.device.metadata.get("iree.driver") for d in ep_devices]
        raise RuntimeError(
            f"IREE device with driver '{iree_driver}' not found. Available: {available}"
        )

    print(f"IREE EP: driver={iree_driver}, device_id={iree_device.device.device_id}")

    sess_options = ort.SessionOptions()
    provider_options = {
        "target_arch": "gfx1100",
        "save_intermediates": "1",
        "opt_level": "O0",
    }
    sess_options.add_provider_for_devices([iree_device], provider_options)

    session = ort.InferenceSession(model_path, sess_options=sess_options)
    return session, iree_device


def get_model_config(session):
    """Extract model configuration from session inputs."""
    model_inputs = session.get_inputs()

    num_layers = 0
    num_kv_heads = 8
    head_dim = 128
    kv_dtype = np.float32  # Default to float32
    has_tree_attention = False
    kv_type_str = None

    input_names = set()
    for inp in model_inputs:
        input_names.add(inp.name)
        if "past_key_values" in inp.name and ".key" in inp.name:
            num_layers += 1
            kv_type_str = inp.type
            if len(inp.shape) == 4:
                if isinstance(inp.shape[1], int):
                    num_kv_heads = inp.shape[1]
                if isinstance(inp.shape[3], int):
                    head_dim = inp.shape[3]
            # Check for dtype - ORT uses "tensor(float16)" or "tensor(float)"
            type_lower = inp.type.lower()
            if "float16" in type_lower or "half" in type_lower:
                kv_dtype = np.float16
            elif "float64" in type_lower or "double" in type_lower:
                kv_dtype = np.float64
            else:
                # "tensor(float)" means float32
                kv_dtype = np.float32

    has_tree_attention = "tree_attention" in input_names

    print(f"Detected KV cache type: {kv_type_str} -> numpy dtype: {kv_dtype}")

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype": kv_dtype,
        "has_tree_attention": has_tree_attention,
        "input_names": input_names,
    }


def generate_text_simple(
    session,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    total_sequence: int = 128,
    window: int = 16,
    context: int = 1024,
):
    """
    Standard HuggingFace ONNX model generation.

    This model expects:
    - position_ids to match input_ids size
    - attention_mask size = past_seq_len + current_seq_len
    """
    config = get_model_config(session)
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    kv_dtype = config["kv_dtype"]
    model_input_names = config["input_names"]

    print(
        f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head dim"
    )
    print(
        f"Model inputs: {sorted([n for n in model_input_names if not n.startswith('past')])}"
    )

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: '{prompt}' -> {prompt_len} tokens")

    output_names = [out.name for out in session.get_outputs()]

    # Initialize empty KV cache
    past_seq_len = 0
    past_key_values = {}
    for i in range(num_layers):
        past_key_values[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_kv_heads, past_seq_len, head_dim), dtype=kv_dtype
        )
        past_key_values[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_kv_heads, past_seq_len, head_dim), dtype=kv_dtype
        )

    generated_ids = input_ids.copy()

    # === PREFILL PHASE ===
    print(f"Prefill phase ({prompt_len} tokens)...")

    # Build inputs based on what the model expects
    inputs = {"input_ids": input_ids}

    if "attention_mask" in model_input_names:
        inputs["attention_mask"] = np.ones((1, prompt_len), dtype=np.int64)

    if "position_ids" in model_input_names:
        inputs["position_ids"] = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)

    # Add KV cache inputs that exist in the model
    for name, value in past_key_values.items():
        if name in model_input_names:
            inputs[name] = value

    outputs = session.run(None, inputs)
    logits = outputs[0]

    # Update KV cache from prefill outputs
    for i, name in enumerate(output_names[1:], start=1):
        past_name = name.replace("present.", "past_key_values.")
        if past_name in past_key_values:
            past_key_values[past_name] = outputs[i]

    past_seq_len = past_key_values[f"past_key_values.0.key"].shape[2]
    print(f"Prefill done, KV cache size: {past_seq_len}")

    # Get first generated token
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
    print(f"First token: {next_token} = '{tokenizer.decode([next_token])}'")

    # === DECODE PHASE ===
    print("Decode phase...")
    for step in range(max_new_tokens - 1):
        total_len = past_seq_len + 1

        # Build inputs based on what the model expects
        inputs = {"input_ids": np.array([[next_token]], dtype=np.int64)}

        if "attention_mask" in model_input_names:
            inputs["attention_mask"] = np.ones((1, total_len), dtype=np.int64)

        if "position_ids" in model_input_names:
            inputs["position_ids"] = np.array([[past_seq_len]], dtype=np.int64)

        # Add KV cache inputs
        for name, value in past_key_values.items():
            if name in model_input_names:
                inputs[name] = value

        outputs = session.run(None, inputs)
        logits = outputs[0]

        # Update KV cache
        for i, name in enumerate(output_names[1:], start=1):
            past_name = name.replace("present.", "past_key_values.")
            if past_name in past_key_values:
                past_key_values[past_name] = outputs[i]

        past_seq_len = past_key_values[f"past_key_values.0.key"].shape[2]

        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)

        if next_token == tokenizer.eos_token_id:
            print(f"EOS at step {step + 1}")
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def generate_text_iobinding(
    session,
    iree_device,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    total_sequence: int = 128,
    window: int = 16,
    context: int = 1024,
):
    """
    ONNX generation with IO binding for IREE that adapts to model inputs.

    Note: IREE doesn't support 0-size tensors, so we initialize KV cache with 1 dummy
    position and use attention masking to ignore it.
    """
    config = get_model_config(session)
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    kv_dtype = config["kv_dtype"]
    model_input_names = config["input_names"]

    print(
        f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head dim"
    )
    print(
        f"Model inputs: {sorted([n for n in model_input_names if not n.startswith('past')])}"
    )

    # Check what optional inputs the model has
    has_attention_mask = "attention_mask" in model_input_names
    has_position_ids = "position_ids" in model_input_names

    device_id = iree_device.device.device_id

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: '{prompt}' -> {prompt_len} tokens")

    output_names = [out.name for out in session.get_outputs()]

    # IREE workaround: Initialize KV cache with 1 dummy position (to avoid 0-size tensors)
    # We'll mask it out with attention_mask (if the model has it)
    num_dummy = 1
    past_key_values = {}
    for i in range(num_layers):
        past_key_values[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_kv_heads, num_dummy, head_dim), dtype=kv_dtype
        )
        past_key_values[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_kv_heads, num_dummy, head_dim), dtype=kv_dtype
        )

    generated_ids = input_ids.copy()

    # === PREFILL PHASE ===
    print(f"Prefill phase ({prompt_len} tokens)...")

    # Build inputs based on what the model expects
    inputs = {"input_ids": input_ids}

    if has_attention_mask:
        # Attention mask: 0 for dummy position, 1 for real tokens
        inputs["attention_mask"] = np.concatenate(
            [
                np.zeros((1, num_dummy), dtype=np.int64),
                np.ones((1, prompt_len), dtype=np.int64),
            ],
            axis=1,
        )

    if has_position_ids:
        # Position IDs: 0, 1, 2, ... for the real tokens (not including dummy)
        inputs["position_ids"] = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)

    # Add KV cache inputs that exist in the model
    for name, value in past_key_values.items():
        if name in model_input_names:
            inputs[name] = value

    # Run prefill
    outputs = session.run(None, inputs)
    logits = outputs[0]

    # Get first generated token
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
    print(f"First token: {next_token} = '{tokenizer.decode([next_token])}'")

    # Update KV cache and move to device
    print("Moving KV cache to device...")
    kv_cache_device = {}
    for i, name in enumerate(output_names[1:], start=1):
        past_name = name.replace("present.", "past_key_values.")
        if "past_key_values" in past_name and past_name in model_input_names:
            kv_data = outputs[i]
            kv_shape = list(kv_data.shape)

            device_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                kv_shape,
                kv_dtype,
                device_type="gpu",
                device_id=device_id,
                vendor_id=IREE_VENDOR_ID,
            )
            device_tensor.update_inplace(kv_data)
            kv_cache_device[past_name] = device_tensor

    # After prefill, KV cache contains: [dummy (1)] + [prompt tokens (prompt_len)]
    # Total KV size = num_dummy + prompt_len
    past_seq_len = num_dummy + prompt_len

    # === DECODE PHASE WITH IO BINDING ===
    print("Decode phase with IO binding...")

    for step in range(max_new_tokens - 1):
        io_binding = session.io_binding()

        # input_ids: single token
        input_ids_tensor = ort.OrtValue.ortvalue_from_numpy(
            np.array([[next_token]], dtype=np.int64)
        )
        io_binding.bind_ortvalue_input("input_ids", input_ids_tensor)

        if has_attention_mask:
            # attention_mask: 0 for dummy, 1 for real tokens + new token
            total_len = past_seq_len + 1
            attention_mask = np.concatenate(
                [
                    np.zeros((1, num_dummy), dtype=np.int64),
                    np.ones((1, total_len - num_dummy), dtype=np.int64),
                ],
                axis=1,
            )
            attn_tensor = ort.OrtValue.ortvalue_from_numpy(attention_mask)
            io_binding.bind_ortvalue_input("attention_mask", attn_tensor)

        if has_position_ids:
            # position_ids: current position (real sequence length, not including dummy)
            real_pos = past_seq_len - num_dummy  # = prompt_len + tokens_generated
            position_ids = np.array([[real_pos]], dtype=np.int64)
            pos_tensor = ort.OrtValue.ortvalue_from_numpy(position_ids)
            io_binding.bind_ortvalue_input("position_ids", pos_tensor)

        # Bind KV cache from device
        for name, tensor in kv_cache_device.items():
            io_binding.bind_ortvalue_input(name, tensor)

        # Bind outputs - use CPU for now to avoid allocator issues
        for name in output_names:
            io_binding.bind_output(name, device_type="cpu")

        # Run
        session.run_with_iobinding(io_binding)
        ort_outputs = io_binding.get_outputs()

        # Get logits
        logits = ort_outputs[0].numpy()
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)

        # Update KV cache on device
        for i, name in enumerate(output_names[1:], start=1):
            past_name = name.replace("present.", "past_key_values.")
            if past_name in kv_cache_device:
                kv_data = ort_outputs[i].numpy()
                kv_shape = list(kv_data.shape)

                device_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                    kv_shape,
                    kv_dtype,
                    device_type="gpu",
                    device_id=device_id,
                    vendor_id=IREE_VENDOR_ID,
                )
                device_tensor.update_inplace(kv_data)
                kv_cache_device[past_name] = device_tensor

        past_seq_len += 1

        if next_token == tokenizer.eos_token_id:
            print(f"EOS at step {step + 1}")
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test Llama 3.1 ONNX model")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to ONNX model")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, help="Path to tokenizer")
    parser.add_argument(
        "--prompt",
        default="The capital of the United States is",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--use-iree", action="store_true", help="Use IREE EP instead of CPU"
    )
    parser.add_argument(
        "--target",
        default="llvm-cpu",
        help="IREE target (llvm-cpu, vulkan-spirv, rocm, etc.)",
    )
    parser.add_argument(
        "--no-iobinding", action="store_true", help="Disable IO binding (for debugging)"
    )
    parser.add_argument(
        "--total-sequence",
        type=int,
        default=128,
        help="Fixed total sequence length for attention",
    )
    parser.add_argument(
        "--window", type=int, default=16, help="Window size for processing tokens"
    )
    parser.add_argument(
        "--context", type=int, default=1024, help="Context size for KV cache"
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading model from {args.model}...")

    iree_device = None
    if args.use_iree:
        print(f"Using IREE EP with target: {args.target}")
        session, iree_device = get_iree_session(args.model, args.target)
    else:
        print("Using CPU provider")
        # Disable memory pattern optimization to allow dynamic shapes
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(
            args.model, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    print_model_info(session)

    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    try:
        gen_kwargs = {
            "max_new_tokens": args.max_tokens,
            "total_sequence": args.total_sequence,
            "window": args.window,
            "context": args.context,
        }

        if args.use_iree and not args.no_iobinding:
            generated = generate_text_iobinding(
                session, iree_device, tokenizer, args.prompt, **gen_kwargs
            )
        else:
            generated = generate_text_simple(
                session, tokenizer, args.prompt, **gen_kwargs
            )

        print(f"\n{'='*60}")
        print(f"Generated text:")
        print(f"{'='*60}")
        print(generated)
        print(f"{'='*60}")

    except Exception as e:
        print(f"\nGeneration failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
