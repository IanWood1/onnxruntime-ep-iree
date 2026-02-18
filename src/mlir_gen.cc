//===- mlir_gen.cc --------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MLIR text generation from OrtGraph.
//
//===----------------------------------------------------------------------===//

#include "mlir_gen.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree_ep.h"
#include "iree_ort_utils.h"

namespace onnxruntime::iree {
namespace {

// Initializers smaller than this are inlined via dense<> DenseElementsAttr.
// Larger ones become IREE parameters backed by an IRPA archive.
constexpr size_t kMaxInlineInitializerSize = 256;

// Encodes raw bytes as a hex string: "0xAABBCC...".
std::string HexEncode(const uint8_t* data, size_t size) {
  constexpr char hex_chars[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 + size * 2);
  result = "0x";
  for (size_t i = 0; i < size; ++i) {
    result += hex_chars[(data[i] >> 4) & 0xF];
    result += hex_chars[data[i] & 0xF];
  }
  return result;
}

// Tracks a large initializer that will become an IREE parameter.
struct ParameterInitializer {
  std::string sanitized_name;
  size_t initializer_index;  // Index into initializers_ vector.
};

// Callback for iree_io_build_parameter_archive to create the IRPA file.
iree_status_t IrpaFileOpenCallback(void* user_data, iree_io_physical_offset_t,
                                   iree_io_physical_size_t,
                                   iree_io_file_handle_t** out_file_handle) {
  auto* path = static_cast<const std::string*>(user_data);
  return iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_OVERWRITE,
      iree_make_string_view(path->data(), path->size()),
      iree_allocator_system(), out_file_handle);
}

// Sanitizes an ONNX name to be a valid MLIR SSA identifier.
// MLIR identifiers must match [a-zA-Z_][a-zA-Z0-9_$]*.
std::string SanitizeName(const std::string& name) {
  assert(!name.empty() && "Unexpected empty name");
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
      result += c;
    } else {
      result += '_';
    }
  }
  // Ensure starts with letter or underscore.
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result.empty() ? "_unnamed" : result;
}

// Joins a vector of strings with a separator.
std::string Join(const std::vector<std::string>& parts,
                 const std::string& sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    ss << parts[i];
  }
  return ss.str();
}

// Returns MLIR element type string for an ONNX tensor element type.
// If signless is true, returns signless types (i64) for all integers.
// Otherwise returns signed (si64) or unsigned (ui64) types for torch dialect.
std::string GetElementType(ONNXTensorElementDataType dtype,
                           bool signless = false) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "f64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "f16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return signless ? "i8" : "si8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return signless ? "i16" : "si16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return signless ? "i32" : "si32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return signless ? "i64" : "si64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return signless ? "i8" : "ui8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return signless ? "i16" : "ui16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return signless ? "i32" : "ui32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return signless ? "i64" : "ui64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "i1";
    default:
      return "NYI";
  }
}

// Formats a tensor type as !torch.vtensor<[dims],dtype>.
// When static_specs is provided, dynamic dims whose symbolic name matches a
// kStatic spec are replaced with the concrete value.
std::string FormatTensorType(
    const Ort::ConstTypeInfo& type_info,
    const std::unordered_map<std::string, const DimSpec*>& static_specs = {},
    const std::vector<const char*>& symbolic_dims = {}) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";  // NYI: non-tensor types.
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "!torch.vtensor<[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    if (shape[i] < 0) {
      // Check if this dynamic dim has a static specialization.
      if (i < symbolic_dims.size() && symbolic_dims[i] != nullptr) {
        auto it = static_specs.find(symbolic_dims[i]);
        if (it != static_specs.end()) {
          ss << it->second->value;
          continue;
        }
      }
      ss << "?";
    } else {
      ss << shape[i];
    }
  }
  ss << "]," << GetElementType(dtype) << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
// When static_specs is provided, dynamic dims whose symbolic name matches a
// kStatic spec are replaced with the concrete value.
std::string FormatMlirTensorType(
    const Ort::ConstTypeInfo& type_info,
    const std::unordered_map<std::string, const DimSpec*>& static_specs = {},
    const std::vector<const char*>& symbolic_dims = {}) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      // Check if this dynamic dim has a static specialization.
      if (i < symbolic_dims.size() && symbolic_dims[i] != nullptr) {
        auto it = static_specs.find(symbolic_dims[i]);
        if (it != static_specs.end()) {
          ss << it->second->value;
          ss << "x";
          continue;
        }
      }
      ss << "?";
    } else {
      ss << shape[i];
    }
    ss << "x";
  }
  ss << GetElementType(dtype, /*signless=*/true) << ">";
  return ss.str();
}

// MLIR generator class.
class MlirGenerator {
 public:
  MlirGenerator(const Ort::ConstGraph& graph, std::ostream& out,
                const std::string& irpa_path, const DimSpecVariant& dim_specs,
                const std::string& function_name_suffix = "")
      : graph_(graph),
        out_(out),
        irpa_path_(irpa_path),
        dim_specs_(dim_specs),
        function_name_suffix_(function_name_suffix) {
    // Build lookup map for static specs by symbolic name.
    for (const auto& spec : dim_specs_) {
      if (spec.kind == DimSpec::Kind::kStatic) {
        static_specs_[spec.symbolic_name] = &spec;
      }
    }
  }

  void Generate() {
    CollectMetadata();
    EmitModuleHeader();
    EmitFunctionBody();
    EmitModuleFooter();
  }

  // Generates a single MLIR module containing multiple functions, one per
  // variant. All functions share the same module (and thus the same parameter
  // references), so when compiled to a single VMFB the weights are shared.
  struct VariantInfo {
    std::string suffix;           // Function name suffix (e.g., "_variant0").
    const DimSpecVariant* specs;  // Dim specs for this variant.
  };

  void GenerateMultiVariant(const std::vector<VariantInfo>& variants) {
    CollectMetadata();
    emit_globals_at_module_scope_ = true;
    out_ << "module {\n";
    EmitModuleScopeGlobals();
    for (const auto& v : variants) {
      ConfigureForVariant(*v.specs, v.suffix);
      EmitFunctionHeader();
      EmitFunctionBody();
      out_ << "  }\n";  // Close function.
    }
    out_ << "}\n";  // Close module.
  }

  // Builds an IRPA parameter archive for large initializers and creates a
  // parameter provider. Call after Generate(). If no parameters are needed,
  // the output pointers remain null.
  OrtStatus* BuildParameterArchive(ParameterIndexPtr& out_index,
                                   ParameterProviderPtr& out_provider);

 private:
  void CollectMetadata() {
    // Get graph name.
    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) {
      graph_name_ = "main";
    }
    graph_name_ += function_name_suffix_;

    // Get IR version.
    ir_version_ = graph_.GetOnnxIRVersion();

    // Get opset version (find default domain).
    auto opsets = graph_.GetOperatorSets();
    for (const auto& opset : opsets) {
      if (opset.domain.empty() || opset.domain == "ai.onnx") {
        opset_version_ = opset.version;
        break;
      }
    }

    // Collect inputs (excluding initializers).
    auto inputs = graph_.GetInputs();
    auto initializers = graph_.GetInitializers();

    // Build set of initializer names.
    std::unordered_set<std::string> init_names;
    for (const auto& init : initializers) {
      init_names.insert(init.GetName());
    }

    // Graph inputs are those not in initializers.
    for (const auto& input : inputs) {
      std::string name = input.GetName();
      if (init_names.find(name) == init_names.end()) {
        graph_inputs_.push_back(input);
      }
    }

    // Graph outputs.
    graph_outputs_ = graph_.GetOutputs();

    // Initializers.
    initializers_ = initializers;

    // Collect symbolic dimension names for graph inputs and outputs.
    // These are used for dim spec matching (static specialization and
    // divisibility constraints).
    for (const auto& input : graph_inputs_) {
      auto type_info = input.TypeInfo();
      if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_symbolic_dims_.push_back(tensor_info.GetSymbolicDimensions());
      } else {
        input_symbolic_dims_.emplace_back();
      }
    }
    for (const auto& output : graph_outputs_) {
      auto type_info = output.TypeInfo();
      if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_symbolic_dims_.push_back(tensor_info.GetSymbolicDimensions());
      } else {
        output_symbolic_dims_.emplace_back();
      }
    }
  }

  // Reconfigures the generator for a new variant within the same module.
  void ConfigureForVariant(const DimSpecVariant& specs,
                           const std::string& suffix) {
    dim_specs_ = specs;
    function_name_suffix_ = suffix;
    static_specs_.clear();
    specialized_types_.clear();
    for (const auto& spec : dim_specs_) {
      if (spec.kind == DimSpec::Kind::kStatic) {
        static_specs_[spec.symbolic_name] = &spec;
      }
    }
    // Recompute graph name with new suffix.
    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) {
      graph_name_ = "main";
    }
    graph_name_ += function_name_suffix_;
  }

  void EmitModuleHeader() {
    out_ << "module {\n";
    EmitFunctionHeader();
  }

  // Emits the function signature with current dim specs and function name.
  void EmitFunctionHeader() {
    // Build function arguments (apply static specialization to signature).
    // Also populate specialized_types_ so node emissions use consistent types.
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      std::string type = FormatTensorType(
          graph_inputs_[i].TypeInfo(), static_specs_, input_symbolic_dims_[i]);
      args << "%" << name << ": " << type;
      if (!static_specs_.empty()) {
        specialized_types_[name] = type;
      }
    }

    // Build return types (apply static specialization to signature).
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      std::string out_name = SanitizeName(graph_outputs_[i].GetName());
      std::string type =
          FormatTensorType(graph_outputs_[i].TypeInfo(), static_specs_,
                           output_symbolic_dims_[i]);
      ret_types << type;
      if (!static_specs_.empty()) {
        specialized_types_[out_name] = type;
      }
    }

    constexpr std::string_view schema = R"(  func.func @{0}({1}) -> ({2})
      attributes {{
        torch.onnx_meta.ir_version = {3} : si64,
        torch.onnx_meta.opset_version = {4} : si64,
        torch.onnx_meta.producer_name = "onnxruntime-ep-iree",
        torch.onnx_meta.producer_version = ""
      }} {{
)";

    out_ << std::format(schema,
                        graph_name_,      // {0}
                        args.str(),       // {1}
                        ret_types.str(),  // {2}
                        ir_version_,      // {3}
                        opset_version_);  // {4}
  }

  // Emits util.global declarations at module scope for large parameter-backed
  // initializers. When multiple functions share a module, this ensures the
  // weights are loaded once and shared across all functions, avoiding OOM from
  // weight duplication.
  void EmitModuleScopeGlobals() {
    for (size_t i = 0; i < initializers_.size(); ++i) {
      const auto& init = initializers_[i];
      std::string name = SanitizeName(init.GetName());
      std::string tensor_type = FormatMlirTensorType(init.TypeInfo());

      auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
      size_t byte_size = tensor_info.GetElementCount() *
                         OnnxElementTypeSize(tensor_info.GetElementType());

      if (byte_size > kMaxInlineInitializerSize) {
        out_ << std::format(
            "  util.global private @__param_{0} = "
            "#flow.parameter.named<\"model\"::\"{0}\"> : {1}\n",
            name, tensor_type);
        parameter_initializers_.push_back({name, i});
      }
    }
  }

  void EmitFunctionBody() {
    // Emit divisibility constraints (torch.symbolic_int + bind_symbolic_shape).
    EmitDivisibilityConstraints();

    // Emit initializers as flow.tensor.constant ops.
    for (size_t i = 0; i < initializers_.size(); ++i) {
      EmitInitializer(initializers_[i], i);
    }

    // Emit nodes.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      EmitNode(node);
    }

    // Emit return.
    EmitReturn();
  }

  // Emits an initializer as a flow.tensor.constant with a
  // torch_c.from_builtin_tensor cast. Small initializers use dense<> with
  // inline hex-encoded data. Large initializers use #flow.parameter.named
  // (data stored in IRPA archive).
  //
  // Output format (small):
  //   %__raw_name = flow.tensor.constant dense<"0x..."> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  //
  // Output format (large):
  //   %__raw_name = flow.tensor.constant
  //       #flow.parameter.named<"model"::"name"> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  void EmitInitializer(const Ort::ConstValueInfo& init, size_t init_index) {
    std::string name = SanitizeName(init.GetName());
    std::string vtensor_type = FormatTensorType(init.TypeInfo());
    std::string tensor_type = FormatMlirTensorType(init.TypeInfo());

    auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
    size_t byte_size = tensor_info.GetElementCount() *
                       OnnxElementTypeSize(tensor_info.GetElementType());

    if (byte_size <= kMaxInlineInitializerSize) {
      // Small: inline with dense<> DenseElementsAttr.
      Ort::ConstValue tensor_value{nullptr};
      auto status = init.GetInitializer(tensor_value);
      if (!status.IsOK()) {
        return;
      }
      const auto* data =
          static_cast<const uint8_t*>(tensor_value.GetTensorRawData());
      std::string hex = HexEncode(data, tensor_value.GetTensorSizeInBytes());

      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant dense<"{3}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type, hex);
    } else if (emit_globals_at_module_scope_) {
      // Multi-variant mode: load from the module-scope util.global.
      // The global was emitted by EmitModuleScopeGlobals() and
      // parameter_initializers_ was populated there.
      constexpr std::string_view schema =
          R"(    %__raw_{0} = util.global.load @__param_{0} : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type);
    } else {
      // Single-function mode: inline parameter reference.
      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant #flow.parameter.named<"model"::"{0}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type);
      parameter_initializers_.push_back({name, init_index});
    }
  }

  void EmitNode(const Ort::ConstNode& node) {
    std::string op_type = node.GetOperatorType();
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Build output SSA names and types.
    // If a node output corresponds to a graph output with a specialized type,
    // use the specialized type to maintain SSA type consistency.
    std::ostringstream out_names;
    std::ostringstream out_types;
    bool first_output = true;
    size_t valid_output_count = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) {
        continue;
      }
      std::string output_name = outputs[i].GetName();
      if (output_name.empty()) {
        continue;
      }
      if (!first_output) {
        out_names << ", ";
        out_types << ", ";
      }
      first_output = false;
      valid_output_count++;
      std::string sanitized = SanitizeName(output_name);
      out_names << "%" << sanitized;
      auto it = specialized_types_.find(sanitized);
      if (it != specialized_types_.end()) {
        out_types << it->second;
      } else {
        out_types << FormatTensorType(outputs[i].TypeInfo());
      }
    }

    // Build input SSA references.
    // If a node input references a graph input with a specialized type,
    // use the specialized type to maintain SSA type consistency.
    std::ostringstream in_names;
    std::ostringstream in_types;
    bool first_input = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) {
        continue;
      }
      std::string input_name = inputs[i].GetName();
      if (input_name.empty()) {
        continue;
      }
      if (!first_input) {
        in_names << ", ";
        in_types << ", ";
      }
      first_input = false;
      std::string sanitized = SanitizeName(input_name);
      in_names << "%" << sanitized;
      auto it = specialized_types_.find(sanitized);
      if (it != specialized_types_.end()) {
        in_types << it->second;
      } else {
        in_types << FormatTensorType(inputs[i].TypeInfo());
      }
    }

    // Build attributes.
    std::string attr_str = FormatAttributes(attrs);

    // Format output types: wrap in parentheses if multiple outputs.
    std::string out_types_str = out_types.str();
    if (valid_output_count > 1 && !out_types_str.empty()) {
      out_types_str = "(" + out_types_str + ")";
    }

    // Emit the operator.
    constexpr std::string_view schema =
        R"(    {0} = torch.operator "onnx.{1}"({2}) {{{3}}} : ({4}) -> {5}
)";
    out_ << std::format(schema,
                        out_names.str(),  // {0}
                        op_type,          // {1}
                        in_names.str(),   // {2}
                        attr_str,         // {3}
                        in_types.str(),   // {4}
                        out_types_str);   // {5}
  }

  std::string FormatAttributes(const std::vector<Ort::ConstOpAttr>& attrs) {
    if (attrs.empty()) {
      return "";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < attrs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << FormatAttribute(attrs[i]);
    }
    return ss.str();
  }

  std::string FormatAttribute(const Ort::ConstOpAttr& attr) {
    std::string name = attr.GetName();
    OrtOpAttrType type = attr.GetType();

    switch (type) {
      case ORT_OP_ATTR_INT: {
        int64_t value = 0;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1} : si64", name, value);
      }
      case ORT_OP_ATTR_FLOAT: {
        float value = 0.0f;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1:e} : f32", name, value);
      }
      case ORT_OP_ATTR_STRING: {
        std::string value;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = \"{1}\"", name, value);
      }
      case ORT_OP_ATTR_INTS: {
        std::vector<int64_t> values;
        attr.GetValueArray<int64_t>(values);
        std::vector<std::string> str_values(values.size());
        std::transform(values.begin(), values.end(), str_values.begin(),
                       [](int64_t v) { return std::format("{0} : si64", v); });
        return std::format("torch.onnx.{0} = [{1}]", name,
                           Join(str_values, ", "));
      }
      default:
        return std::format("torch.onnx.{0} = \"NYI\"", name);
    }
  }

  void EmitReturn() {
    std::ostringstream ret_values;
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_values << ", ";
        ret_types << ", ";
      }
      ret_values << "%" << SanitizeName(graph_outputs_[i].GetName());
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo(), static_specs_,
                                    output_symbolic_dims_[i]);
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
  }

  // Emits torch.symbolic_int and torch.bind_symbolic_shape ops for
  // kDivisibleBy dim specs. This tells the compiler that certain dynamic
  // dimensions are multiples of a given divisor.
  void EmitDivisibilityConstraints() {
    // Collect divisibility specs.
    std::unordered_map<std::string, const DimSpec*> div_specs;
    for (const auto& spec : dim_specs_) {
      if (spec.kind == DimSpec::Kind::kDivisibleBy) {
        div_specs[spec.symbolic_name] = &spec;
      }
    }
    if (div_specs.empty()) {
      return;
    }

    // Collect ALL unique symbolic dim names across graph inputs (both
    // constrained and unconstrained). Each gets a torch.symbolic_int so
    // that unconstrained dims are independent symbols in affine maps.
    struct SymInfo {
      std::string ssa_name;   // e.g., %_sym_0
      std::string sym_label;  // e.g., s_sequence_length
      int64_t divisor;        // 0 = unconstrained, >0 = divisibility
    };
    std::unordered_map<std::string, SymInfo> sym_infos;
    size_t sym_counter = 0;

    // First, register constrained dims.
    for (const auto& [name, spec] : div_specs) {
      SymInfo info;
      info.ssa_name = std::format("%_sym_{}", sym_counter);
      info.sym_label = "s_" + SanitizeName(name);
      info.divisor = spec->value;
      sym_infos[name] = info;
      sym_counter++;
    }

    // Then, register ALL unconstrained dynamic dims from ALL inputs.
    // Every input gets a bind_symbolic_shape so the compiler has
    // consistent shape info across all tensors (matching how torch.export
    // generates these ops).
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      auto type_info = graph_inputs_[i].TypeInfo();
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        continue;
      }
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];

      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] >= 0) continue;  // Static dim.
        if (d < sym_dims.size() && sym_dims[d] != nullptr) {
          std::string name = sym_dims[d];
          if (sym_infos.find(name) == sym_infos.end() &&
              static_specs_.find(name) == static_specs_.end()) {
            // Unconstrained dim with a symbolic name — give it its own symbol.
            SymInfo info;
            info.ssa_name = std::format("%_sym_{}", sym_counter);
            info.sym_label = "s_" + SanitizeName(name);
            info.divisor = 0;
            sym_infos[name] = info;
            sym_counter++;
          }
        }
      }
    }

    // Emit torch.symbolic_int declarations.
    // Sort by symbol index for deterministic output.
    std::vector<std::pair<std::string, SymInfo*>> sorted_syms;
    for (auto& [name, info] : sym_infos) {
      sorted_syms.emplace_back(name, &info);
    }
    std::sort(sorted_syms.begin(), sorted_syms.end(),
              [](const auto& a, const auto& b) {
                return a.second->ssa_name < b.second->ssa_name;
              });
    for (const auto& [name, info] : sorted_syms) {
      out_ << std::format(
          "    {0} = torch.symbolic_int \"{1}\" "
          "{{min_val = 1, max_val = 100000}} : !torch.int\n",
          info->ssa_name, info->sym_label);
    }

    // For each graph input that has at least one dynamic dim with a known
    // symbolic name, emit torch.bind_symbolic_shape. This binds ALL inputs
    // (not just constrained ones) so the compiler has consistent shape info.
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      auto type_info = graph_inputs_[i].TypeInfo();
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        continue;
      }
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];

      // Collect which symbolic ints this input references.
      std::vector<std::string> sym_ssa_list;
      std::unordered_set<std::string> seen_names;
      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] < 0 && d < sym_dims.size() && sym_dims[d] != nullptr) {
          std::string name = sym_dims[d];
          auto it = sym_infos.find(name);
          if (it != sym_infos.end() &&
              seen_names.find(name) == seen_names.end()) {
            sym_ssa_list.push_back(it->second.ssa_name);
            seen_names.insert(name);
          }
        }
      }

      // Skip inputs with no referenced symbolic dims (fully static tensors).
      if (sym_ssa_list.empty()) {
        continue;
      }

      // Build a local mapping from symbolic name to s-index for this input's
      // affine map. The s-index corresponds to position in sym_ssa_list.
      std::unordered_map<std::string, size_t> local_s_index;
      {
        size_t idx = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
          if (shape[d] < 0 && d < sym_dims.size() && sym_dims[d] != nullptr) {
            std::string name = sym_dims[d];
            if (sym_infos.count(name) &&
                local_s_index.find(name) == local_s_index.end()) {
              local_s_index[name] = idx++;
            }
          }
        }
      }

      // Build the affine map expression.
      std::ostringstream affine_params;
      affine_params << "()[";
      for (size_t j = 0; j < sym_ssa_list.size(); ++j) {
        if (j > 0) affine_params << ", ";
        affine_params << "s" << j;
      }
      affine_params << "]";

      std::ostringstream affine_exprs;
      affine_exprs << "(";
      for (size_t d = 0; d < shape.size(); ++d) {
        if (d > 0) affine_exprs << ", ";
        if (shape[d] >= 0) {
          // Static dim: emit constant.
          affine_exprs << shape[d];
        } else if (d < sym_dims.size() && sym_dims[d] != nullptr &&
                   static_specs_.count(sym_dims[d])) {
          // Statically specialized dim: emit the concrete value.
          affine_exprs << static_specs_.at(sym_dims[d])->value;
        } else if (d < sym_dims.size() && sym_dims[d] != nullptr) {
          std::string name = sym_dims[d];
          auto it = sym_infos.find(name);
          if (it != sym_infos.end()) {
            size_t s_idx = local_s_index[name];
            if (it->second.divisor > 0) {
              // Constrained dim: s_i * divisor.
              affine_exprs << "s" << s_idx << " * " << it->second.divisor;
            } else {
              // Unconstrained dim with its own symbol: just s_i.
              affine_exprs << "s" << s_idx;
            }
          } else {
            // Should not happen since we registered all named dims above.
            affine_exprs << "s0";
          }
        } else {
          // Dynamic dim with no symbolic name — shouldn't appear in practice
          // since we only process inputs with constraints. Emit s0 as
          // fallback.
          affine_exprs << "s0";
        }
      }
      affine_exprs << ")";

      std::string input_name = SanitizeName(graph_inputs_[i].GetName());
      // Use the specialized type if available (must match function signature).
      auto spec_it = specialized_types_.find(input_name);
      std::string vtensor_type =
          spec_it != specialized_types_.end()
              ? spec_it->second
              : FormatTensorType(graph_inputs_[i].TypeInfo());

      out_ << std::format(
          "    torch.bind_symbolic_shape %{0}, [{1}], "
          "affine_map<{2} -> {3}> : {4}\n",
          input_name, Join(sym_ssa_list, ", "), affine_params.str(),
          affine_exprs.str(), vtensor_type);
    }
  }

  void EmitModuleFooter() {
    out_ << "  }\n";  // Close function.
    out_ << "}\n";    // Close module.
  }

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;
  std::string irpa_path_;
  DimSpecVariant dim_specs_;
  bool emit_globals_at_module_scope_ = false;

  // Lookup map: symbolic_name -> DimSpec* for kStatic specs.
  std::unordered_map<std::string, const DimSpec*> static_specs_;

  // Map from sanitized SSA name -> specialized vtensor type string.
  // Populated in EmitModuleHeader for graph inputs/outputs that have static
  // specializations. Used by EmitNode to maintain SSA type consistency.
  std::unordered_map<std::string, std::string> specialized_types_;

  std::string graph_name_;
  std::string function_name_suffix_;
  int64_t ir_version_ = 8;
  int64_t opset_version_ = 17;

  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstValueInfo> initializers_;
  std::vector<ParameterInitializer> parameter_initializers_;

  // Symbolic dimension names per graph input/output (parallel to
  // graph_inputs_/graph_outputs_).
  std::vector<std::vector<const char*>> input_symbolic_dims_;
  std::vector<std::vector<const char*>> output_symbolic_dims_;
};

// Builds an IRPA parameter archive for large initializers.
//
// Large inline initializers are copied into an IRPA (IREE Parameter Archive)
// file on disk. We write to an IRPA file rather than keeping data in memory
// because ORT does not guarantee that initializer tensor data remains valid
// beyond the Compile() call. By persisting to disk via IRPA, the data is
// accessed at runtime through IREE's parameter index with file-backed entries.
//
// External initializers (already backed by external files) are added to the
// parameter index directly, pointing to their original files without copying.
//
// The resulting parameter provider is registered with the IREE session so that
// the compiled module can resolve #flow.parameter.named references at runtime.
OrtStatus* MlirGenerator::BuildParameterArchive(
    ParameterIndexPtr& out_index, ParameterProviderPtr& out_provider) {
  if (parameter_initializers_.empty()) {
    return nullptr;
  }

  iree_allocator_t allocator = iree_allocator_system();

  // Build source index from ORT tensor data wrapped as file handles.
  // The tensor data is valid for the duration of this call (we are inside
  // CompileImpl). iree_io_build_parameter_archive copies it to the IRPA file.
  ParameterIndexPtr source_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, source_index.ForOutput()));

  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    // Skip external initializers — added to target index later.
    // Note: GetExternalInitializerInfo returns OK with null output for
    // non-external initializers, so we must check both status and pointer.
    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (ext_info) {
      continue;
    }

    Ort::ConstValue tensor(nullptr);
    auto status = init.GetInitializer(tensor);
    if (!status.IsOK()) {
      return Ort::Status(
                 std::format("Failed to get initializer: {}", init.GetName())
                     .c_str(),
                 ORT_FAIL)
          .release();
    }

    auto* data = const_cast<uint8_t*>(
        static_cast<const uint8_t*>(tensor.GetTensorRawData()));
    size_t size = tensor.GetTensorSizeInBytes();

    FileHandlePtr handle;
    iree_byte_span_t span = {data, static_cast<iree_host_size_t>(size)};
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ, span,
        iree_io_file_handle_release_callback_null(), allocator,
        handle.ForOutput()));

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = size;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = handle.Get();
    entry.storage.file.offset = 0;
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(source_index.Get(), &entry));
  }

  // Build IRPA archive from source index.
  ParameterIndexPtr target_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, target_index.ForOutput()));

  if (iree_io_parameter_index_count(source_index.Get()) > 0) {
    iree_io_parameter_archive_file_open_callback_t file_open = {
        IrpaFileOpenCallback,
        const_cast<std::string*>(&irpa_path_),
    };
    IREE_ORT_RETURN_IF_ERROR(iree_io_build_parameter_archive(
        source_index.Get(), target_index.Get(), file_open, 0, allocator));
  }

  // Add external initializer entries directly to target index.
  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (!ext_info) {
      continue;
    }

    FileHandlePtr ext_handle;
    // External data paths are relative to the model directory.
    std::filesystem::path model_dir =
        std::filesystem::path(graph_.GetModelPath()).parent_path();
    std::string filepath = (model_dir / ext_info.GetFilePath()).string();
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ,
        iree_make_string_view(filepath.data(), filepath.size()), allocator,
        ext_handle.ForOutput()));

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = ext_info.GetByteSize();
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = ext_handle.Get();
    entry.storage.file.offset = static_cast<uint64_t>(ext_info.GetFileOffset());
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(target_index.Get(), &entry));
  }

  ParameterProviderPtr provider;
  IREE_ORT_RETURN_IF_ERROR(iree_io_parameter_index_provider_create(
      iree_make_cstring_view("model"), target_index.Get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      allocator, provider.ForOutput()));

  out_index = std::move(target_index);
  out_provider = std::move(provider);
  return nullptr;
}

}  // namespace

OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
                        const std::string& mlir_path,
                        const std::string& irpa_path,
                        const DimSpecVariant& dim_specs,
                        ParameterIndexPtr& out_index,
                        ParameterProviderPtr& out_provider, bool build_irpa,
                        const std::string& function_name_suffix) {
  std::ofstream file(mlir_path);
  if (!file.is_open()) {
    return Ort::Status(
               std::format("Failed to open output file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  MlirGenerator gen(graph, file, irpa_path, dim_specs, function_name_suffix);
  gen.Generate();

  file.close();
  if (file.fail()) {
    return Ort::Status(
               std::format("Failed to write to file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  if (build_irpa) {
    return gen.BuildParameterArchive(out_index, out_provider);
  }
  return nullptr;
}

OrtStatus* GenerateMultiVariantMlir(
    const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
    const std::string& mlir_path, const std::string& irpa_path,
    const std::vector<std::pair<std::string, DimSpecVariant>>& variants,
    ParameterIndexPtr& out_index, ParameterProviderPtr& out_provider) {
  std::ofstream file(mlir_path);
  if (!file.is_open()) {
    return Ort::Status(
               std::format("Failed to open output file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  // Use empty specs for initial construction; GenerateMultiVariant reconfigures
  // per function.
  DimSpecVariant empty;
  MlirGenerator gen(graph, file, irpa_path, empty);

  std::vector<MlirGenerator::VariantInfo> infos;
  infos.reserve(variants.size());
  for (const auto& [suffix, specs] : variants) {
    infos.push_back({suffix, &specs});
  }
  gen.GenerateMultiVariant(infos);

  file.close();
  if (file.fail()) {
    return Ort::Status(
               std::format("Failed to write to file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  return gen.BuildParameterArchive(out_index, out_provider);
}

}  // namespace onnxruntime::iree
