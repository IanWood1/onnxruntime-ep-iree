add_library(iree_ep_runtime OBJECT iree_ort_utils.cc)
add_library(iree_ep::runtime ALIAS iree_ep_runtime)

target_link_libraries(iree_ep_runtime PUBLIC iree_ep::utils)
target_compile_options(iree_ep_runtime PRIVATE ${IREE_EP_COMPILE_OPTIONS})
