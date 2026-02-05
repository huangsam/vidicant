cmake_minimum_required(VERSION 3.24)

project(
  vidicant
  VERSION 1.0
  LANGUAGES CXX
)

# Define C++ standard and export compile commands
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch pybind11 automatically
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(pybind11)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

# Define library target
add_library(vidicant_lib 
  src/image.cpp
  src/video.cpp
)

# Set position-independent code for static library to be linked into shared objects
set_target_properties(vidicant_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Include directories and link libraries
target_include_directories(vidicant_lib PRIVATE include)
target_link_libraries(vidicant_lib PRIVATE ${OpenCV_LIBS})

# Add executable target for CLI
add_executable(vidicant_cli src/main.cpp src/controller.cpp)
target_include_directories(vidicant_cli PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vidicant_cli PRIVATE vidicant_lib)

# Apply target-specific compile options
target_compile_options(vidicant_cli PRIVATE 
  -Wall 
  -Wextra 
  -Wpedantic 
  -Wunused 
  -Werror
)

# Create Python extension module using pybind11
pybind11_add_module(vidicant_py 
  src/vidicant_py.cpp 
  src/controller.cpp
)
target_include_directories(vidicant_py PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vidicant_py PRIVATE vidicant_lib ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

# Install Python extension module
install(TARGETS vidicant_py DESTINATION vidicant)

# Enable testing and add test subdirectory
enable_testing()
add_subdirectory(test)
