cmake_minimum_required(VERSION 3.15)
project(daedalus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(daedalus_cpp 
    src/bindings/bindings.cc 
    src/models/linearRegression.cc
    src/models/logisticRegression.cc
    src/models/knn.cc
    src/models/DenseLayer.cc
    src/models/NeuralNetwork.cc
)

set_target_properties(daedalus_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/daedalus"
)

target_include_directories(daedalus_cpp PRIVATE include)

if(WIN32)
    target_link_options(daedalus_cpp PRIVATE "-static-libgcc" "-static-libstdc++")
endif()

install(TARGETS daedalus_cpp LIBRARY DESTINATION daedalus)

find_package(Python COMPONENTS Interpreter REQUIRED)

# Stub generation setup
set(STUB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/daedalus")

add_custom_command(
    TARGET daedalus_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:daedalus_cpp>" 
            ${Python_EXECUTABLE} -m pybind11_stubgen daedalus_cpp 
            -o "${STUB_SOURCE_DIR}"
    COMMENT "Generating stubs into the source directory for editable installs"
)