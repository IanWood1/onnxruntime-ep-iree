add_library(iree_ep_utils OBJECT temp_file.cc)
add_library(iree_ep::utils ALIAS iree_ep_utils)

target_include_directories(iree_ep_utils PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(iree_ep_utils SYSTEM PUBLIC ${IREE_EP_ONNX_SRC_DIR}/include)
target_link_libraries(iree_ep_utils PUBLIC iree_runtime_unified)
target_compile_features(iree_ep_utils PUBLIC cxx_std_20)
target_compile_options(iree_ep_utils PRIVATE ${IREE_EP_COMPILE_OPTIONS})
