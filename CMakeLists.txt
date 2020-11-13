cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(celeba)

find_package(Torch REQUIRED)

option(DOWNLOAD_CELEBA "Download the CELEBA dataset from the internet" ON)
if (DOWNLOAD_CELEBA)
  message(STATUS "Downloading CELEBA dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/utils/download.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading CELEBA dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(celeba celeba.cpp)
target_link_libraries(celeba "${TORCH_LIBRARIES}")
set_property(TARGET celeba PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET celeba
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:celeba>)
endif (MSVC)