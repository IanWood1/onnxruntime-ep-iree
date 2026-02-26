//===- ep_defs.h ----------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EP constants and shared type definitions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_UTILS_EP_DEFS_H_
#define ONNXRUNTIME_EP_IREE_SRC_UTILS_EP_DEFS_H_

#include <cstdint>

#include "utils/ort_import.h"

namespace onnxruntime::iree {

// EP configuration constants
inline constexpr const char* kEpVendor = "IREE";
inline constexpr uint32_t kEpVendorId = 0x1EEE;  // "IREE" in hex-ish
inline constexpr const char* kEpVersion = "0.1.0";

// Hardware vendor IDs for device matching.
// These match OrtDevice::VendorIds from onnxruntime/core/framework/ortdevice.h.
namespace VendorIds {
inline constexpr uint32_t kAmd = 0x1002;     // AMD: ROCm, MIGraphX EPs
inline constexpr uint32_t kNvidia = 0x10DE;  // NVIDIA: CUDA/TensorRT
inline constexpr uint32_t kIntel = 0x8086;   // Intel: OpenVINO
}  // namespace VendorIds

// Helper struct to pass API pointers
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_UTILS_EP_DEFS_H_
