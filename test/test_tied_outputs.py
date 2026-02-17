#!/usr/bin/env python3
"""Test mutable output tensors via DPS (destination-passing style).

When an ONNX output name matches an input name after replacing "present."
with "past_key_values.", the output is tied to that input. IREE writes results
in-place via torch.overwrite.tensor.contents, and only non-tied outputs appear
in the IREE function return list.
"""

import pathlib
import sys

import numpy as np
from onnx import TensorProto, helper

import test_utils


def test_tied_outputs():
    """Test that tied outputs (present.* -> past_key_values.*) work correctly.

    Builds a model with:
    - input_x: a normal input
    - past_key_values.0.key: a KV cache input (becomes mutable via DPS)
    - logits: a normal output (Add of input_x and a constant)
    - present.0.key: a tied output (Add of past_key_values.0.key and a constant)

    The tied output should be written in-place to the mutable input buffer.
    The normal output should be returned from the IREE function as usual.
    """
    device = test_utils.get_iree_device("local-task")
    if not device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    # Shapes
    x_shape = [2, 4]
    kv_shape = [1, 2, 4, 4]

    # Inputs
    input_x = helper.make_tensor_value_info("input_x", TensorProto.FLOAT, x_shape)
    past_kv = helper.make_tensor_value_info(
        "past_key_values.0.key", TensorProto.FLOAT, kv_shape
    )

    # Outputs: logits (normal) + present.0.key (tied to past_key_values.0.key)
    output_logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, x_shape)
    output_present = helper.make_tensor_value_info(
        "present.0.key", TensorProto.FLOAT, kv_shape
    )

    # Constants to add
    logits_const_data = np.ones(x_shape, dtype=np.float32)
    kv_const_data = np.full(kv_shape, 2.0, dtype=np.float32)

    logits_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["logits_bias"],
        value=helper.make_tensor(
            name="logits_bias_val",
            data_type=TensorProto.FLOAT,
            dims=x_shape,
            vals=logits_const_data.flatten().tolist(),
        ),
    )
    kv_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["kv_bias"],
        value=helper.make_tensor(
            name="kv_bias_val",
            data_type=TensorProto.FLOAT,
            dims=kv_shape,
            vals=kv_const_data.flatten().tolist(),
        ),
    )

    # logits = input_x + logits_bias
    add_logits = helper.make_node(
        "Add", inputs=["input_x", "logits_bias"], outputs=["logits"]
    )
    # present.0.key = past_key_values.0.key + kv_bias
    add_kv = helper.make_node(
        "Add",
        inputs=["past_key_values.0.key", "kv_bias"],
        outputs=["present.0.key"],
    )

    graph = helper.make_graph(
        [logits_const, kv_const, add_logits, add_kv],
        "test_tied",
        [input_x, past_kv],
        [output_logits, output_present],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    model_path = test_utils.save_model(model)
    try:
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "save_intermediates": "1"}
        )

        # Run inference
        x_data = np.random.rand(*x_shape).astype(np.float32)
        kv_data = np.random.rand(*kv_shape).astype(np.float32)

        outputs = session.run(
            None,
            {
                "input_x": x_data,
                "past_key_values.0.key": kv_data,
            },
        )

        if len(outputs) != 2:
            print(f"FAIL: Expected 2 outputs, got {len(outputs)}")
            return False

        # Check logits = input_x + 1.0
        expected_logits = x_data + logits_const_data
        if not np.allclose(outputs[0], expected_logits, rtol=1e-5, atol=1e-5):
            print("FAIL: logits mismatch")
            print(f"  Expected: {expected_logits.flatten()[:4]}...")
            print(f"  Got:      {outputs[0].flatten()[:4]}...")
            return False
        print("Normal output (logits) verified!")

        # Check present.0.key = past_key_values.0.key + 2.0
        expected_kv = kv_data + kv_const_data
        if not np.allclose(outputs[1], expected_kv, rtol=1e-5, atol=1e-5):
            print("FAIL: present.0.key mismatch")
            print(f"  Expected: {expected_kv.flatten()[:4]}...")
            print(f"  Got:      {outputs[1].flatten()[:4]}...")
            return False
        print("Tied output (present.0.key) verified!")

        print(f"  input_x shape:              {x_data.shape}")
        print(f"  past_key_values.0.key shape: {kv_data.shape}")
        print(f"  logits shape:               {outputs[0].shape}")
        print(f"  present.0.key shape:        {outputs[1].shape}")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== Test PASSED ===")
    return True


if __name__ == "__main__":
    test_utils.register_ep()
    success = test_tied_outputs()
    sys.exit(0 if success else 1)
