#!/usr/bin/env python3
"""Test that ALL outputs use DPS (destination-passing style).

Every graph output should get a mutable output parameter in the IREE function
signature. The IREE function should return nothing — all results are written
in-place via torch.overwrite.tensor.contents.

This covers:
1. Models where all outputs match KV cache naming (previously broken: invalid
   MLIR `return :` when all outputs were tied).
2. Models with normal output names (no KV cache convention) — these should
   also be DPS.
3. Mixed models with both normal and KV-cache-style outputs.
"""

import pathlib
import sys

import numpy as np
from onnx import TensorProto, helper

import test_utils


def test_all_outputs_tied():
    """Test a model where ALL outputs have present.*/past_key_values.* naming.

    This previously failed because EmitReturn() produced invalid MLIR
    `return  : ` when zero non-tied outputs remained.

    Model:
    - Inputs: past_key_values.0.key, past_key_values.0.value
    - Outputs: present.0.key, present.0.value (both tied to inputs)
    """
    device = test_utils.get_iree_device("local-task")
    if not device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    kv_key_shape = [1, 2, 4, 4]
    kv_value_shape = [1, 2, 4, 4]

    # Inputs
    past_key = helper.make_tensor_value_info(
        "past_key_values.0.key", TensorProto.FLOAT, kv_key_shape
    )
    past_value = helper.make_tensor_value_info(
        "past_key_values.0.value", TensorProto.FLOAT, kv_value_shape
    )

    # Outputs (tied to inputs via present.* -> past_key_values.* convention)
    present_key = helper.make_tensor_value_info(
        "present.0.key", TensorProto.FLOAT, kv_key_shape
    )
    present_value = helper.make_tensor_value_info(
        "present.0.value", TensorProto.FLOAT, kv_value_shape
    )

    # Constants to add
    key_bias_data = np.full(kv_key_shape, 1.0, dtype=np.float32)
    value_bias_data = np.full(kv_value_shape, 2.0, dtype=np.float32)

    key_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["key_bias"],
        value=helper.make_tensor(
            name="key_bias_val",
            data_type=TensorProto.FLOAT,
            dims=kv_key_shape,
            vals=key_bias_data.flatten().tolist(),
        ),
    )
    value_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["value_bias"],
        value=helper.make_tensor(
            name="value_bias_val",
            data_type=TensorProto.FLOAT,
            dims=kv_value_shape,
            vals=value_bias_data.flatten().tolist(),
        ),
    )

    # present.0.key = past_key_values.0.key + key_bias
    add_key = helper.make_node(
        "Add",
        inputs=["past_key_values.0.key", "key_bias"],
        outputs=["present.0.key"],
    )
    # present.0.value = past_key_values.0.value + value_bias
    add_value = helper.make_node(
        "Add",
        inputs=["past_key_values.0.value", "value_bias"],
        outputs=["present.0.value"],
    )

    graph = helper.make_graph(
        [key_const, value_const, add_key, add_value],
        "test_all_tied",
        [past_key, past_value],
        [present_key, present_value],
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

        key_data = np.random.rand(*kv_key_shape).astype(np.float32)
        value_data = np.random.rand(*kv_value_shape).astype(np.float32)

        outputs = session.run(
            None,
            {
                "past_key_values.0.key": key_data,
                "past_key_values.0.value": value_data,
            },
        )

        if len(outputs) != 2:
            print(f"FAIL: Expected 2 outputs, got {len(outputs)}")
            return False

        expected_key = key_data + key_bias_data
        if not np.allclose(outputs[0], expected_key, rtol=1e-5, atol=1e-5):
            print("FAIL: present.0.key mismatch")
            return False
        print("present.0.key verified!")

        expected_value = value_data + value_bias_data
        if not np.allclose(outputs[1], expected_value, rtol=1e-5, atol=1e-5):
            print("FAIL: present.0.value mismatch")
            return False
        print("present.0.value verified!")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== test_all_outputs_tied PASSED ===")
    return True


def test_normal_outputs_dps():
    """Test that normal outputs (no KV cache naming) also use DPS.

    Model:
    - Input: x
    - Outputs: y (= x + 1), z (= x * 2)
    - Neither output matches any KV cache naming convention.
    - Both should still be DPS.
    """
    device = test_utils.get_iree_device("local-task")
    if not device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    shape = [3, 4]

    input_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    output_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    output_z = helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)

    add_const_data = np.ones(shape, dtype=np.float32)
    mul_const_data = np.full(shape, 2.0, dtype=np.float32)

    add_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["add_bias"],
        value=helper.make_tensor(
            name="add_bias_val",
            data_type=TensorProto.FLOAT,
            dims=shape,
            vals=add_const_data.flatten().tolist(),
        ),
    )
    mul_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mul_factor"],
        value=helper.make_tensor(
            name="mul_factor_val",
            data_type=TensorProto.FLOAT,
            dims=shape,
            vals=mul_const_data.flatten().tolist(),
        ),
    )

    add_node = helper.make_node("Add", inputs=["x", "add_bias"], outputs=["y"])
    mul_node = helper.make_node("Mul", inputs=["x", "mul_factor"], outputs=["z"])

    graph = helper.make_graph(
        [add_const, mul_const, add_node, mul_node],
        "test_normal_dps",
        [input_x],
        [output_y, output_z],
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

        x_data = np.random.rand(*shape).astype(np.float32)
        outputs = session.run(None, {"x": x_data})

        if len(outputs) != 2:
            print(f"FAIL: Expected 2 outputs, got {len(outputs)}")
            return False

        expected_y = x_data + add_const_data
        if not np.allclose(outputs[0], expected_y, rtol=1e-5, atol=1e-5):
            print("FAIL: y mismatch")
            return False
        print("Output y (x + 1) verified!")

        expected_z = x_data * mul_const_data
        if not np.allclose(outputs[1], expected_z, rtol=1e-5, atol=1e-5):
            print("FAIL: z mismatch")
            return False
        print("Output z (x * 2) verified!")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== test_normal_outputs_dps PASSED ===")
    return True


if __name__ == "__main__":
    test_utils.register_ep()
    all_passed = True
    for test_fn in [test_all_outputs_tied, test_normal_outputs_dps]:
        print(f"\n--- Running {test_fn.__name__} ---")
        if not test_fn():
            all_passed = False
    sys.exit(0 if all_passed else 1)
