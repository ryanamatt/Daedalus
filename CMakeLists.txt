cmake_minimum_required(VERSION 3.15)
project(daedalus LANGUAGES CXX)

# --- Versioning ---
if(SKBUILD_PROJECT_VERSION)
    set(FULL_VERSION_STR ${SKBUILD_PROJECT_VERSION})
else()
    set(FULL_VERSION_STR "0.0.0-dev")
endif()

add_definitions(-DDAEDALUS_VERSION="${FULL_VERSION_STR}")

# --- Standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
# Enable modern Python discovery to avoid CMP0148 warnings in 3.14+
set(PYBIND11_FINDPYTHON ON) 
find_package(pybind11 REQUIRED)

# --- Project Target ---
pybind11_add_module(daedalus_cpp 
    src/bindings/bindings.cc 
    src/models/linearRegression.cc
    src/models/logisticRegression.cc
    src/models/knn.cc
    src/models/DenseLayer.cc
    src/models/NeuralNetwork.cc
    src/optimization/SimplexSolver.cc
)

set_target_properties(daedalus_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
)

target_include_directories(daedalus_cpp PRIVATE include)

# --- Compiler Specifics ---
# Only apply static linking for MinGW/GCC; MSVC doesn't recognize these flags
if(MINGW OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_options(daedalus_cpp PRIVATE "-static-libgcc" "-static-libstdc++")
endif()

install(TARGETS daedalus_cpp LIBRARY DESTINATION daedalus)

# --- Stub Generation Setup ---
find_package(Python COMPONENTS Interpreter REQUIRED)

set(STUB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/daedalus")

# Using Python_EXECUTABLE ensures we use the interpreter provided by scikit-build
# which contains the build-dependencies (pybind11-stubgen) 
add_custom_command(
    TARGET daedalus_cpp POST_BUILD
    COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen daedalus_cpp 
            -o "${STUB_SOURCE_DIR}"
    # We remove the hardcoded PYTHONPATH env to prevent shadowing the build environment's site-packages
    WORKING_DIRECTORY "${STUB_SOURCE_DIR}"
    COMMENT "Generating stubs into the source directory for editable installs"
)