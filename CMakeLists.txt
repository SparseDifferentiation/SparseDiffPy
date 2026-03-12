cmake_minimum_required(VERSION 3.15)
project(SparseDiffPy LANGUAGES C)

add_subdirectory(SparseDiffEngine)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Linux: CMake find_package(BLAS) doesn't set include dirs;
# openblas puts cblas.h in /usr/include/openblas/
if(UNIX AND NOT APPLE)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas /usr/include)
    if(OPENBLAS_INCLUDE_DIR)
        target_include_directories(dnlp_diff PRIVATE ${OPENBLAS_INCLUDE_DIR})
    endif()
endif()

python3_add_library(_sparsediffengine MODULE
    sparsediffpy/_bindings/bindings.c
)

target_include_directories(_sparsediffengine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseDiffEngine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseDiffEngine/src
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsediffpy/_bindings
)

target_link_libraries(_sparsediffengine PRIVATE dnlp_diff Python3::NumPy)

install(TARGETS _sparsediffengine DESTINATION sparsediffpy)
