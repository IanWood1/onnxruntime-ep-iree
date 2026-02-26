add_library(iree_ep_mlir OBJECT mlir_gen.cc iree_compile.cc)
add_library(iree_ep::mlir ALIAS iree_ep_mlir)

target_link_libraries(iree_ep_mlir PUBLIC
  iree_ep::runtime
  iree_io_parameter_index
  iree_io_parameter_index_provider
  iree_io_formats_irpa_irpa
)
target_compile_options(iree_ep_mlir PRIVATE ${IREE_EP_COMPILE_OPTIONS})
