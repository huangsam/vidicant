cmake_minimum_required(VERSION 3.24)

project(
  vidicant
  VERSION 1.0
  LANGUAGES CXX
)

# 1. Standard and Compile Commands
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Dependencies
find_package(OpenCV REQUIRED)

# 3. Target Definition
add_executable(vidicant_cli 
  src/main.cpp 
  src/hello.cpp
  src/image.cpp
)

# 4. Target-Specific Settings (The Modern Way)
target_include_directories(vidicant_cli PRIVATE include)

target_compile_options(vidicant_cli PRIVATE 
  -Wall 
  -Wextra 
  -Wpedantic 
  -Wunused 
  -Werror
)

# 5. Linking (OpenCV handles its own includes automatically here)
target_link_libraries(vidicant_cli PRIVATE ${OpenCV_LIBS})
