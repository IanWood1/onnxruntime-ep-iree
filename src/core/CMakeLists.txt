add_library(iree_ep_core OBJECT
  plugin_entry.cc
  iree_ep_factory.cc
  iree_ep.cc
  iree_allocator.cc
  iree_data_transfer.cc
)
add_library(iree_ep::core ALIAS iree_ep_core)

target_link_libraries(iree_ep_core PUBLIC
  iree_ep::mlir
  iree_ep::runtime
  iree_modules_io_parameters_parameters
)
target_compile_options(iree_ep_core PRIVATE ${IREE_EP_COMPILE_OPTIONS})
