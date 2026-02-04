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

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

# Define library target
add_library(vidicant_lib 
  src/image.cpp
  src/video.cpp
)

# Include directories and link libraries
target_include_directories(vidicant_lib PRIVATE include)
target_link_libraries(vidicant_lib PRIVATE ${OpenCV_LIBS})

# Add executable target for CLI
add_executable(vidicant_cli src/main.cpp)
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

# Enable testing and add test subdirectory
enable_testing()
add_subdirectory(test)
