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
find_package(GTest REQUIRED)

# 3. Library
add_library(vidicant_lib 
  src/hello.cpp
  src/image.cpp
)

target_include_directories(vidicant_lib PRIVATE include)
target_link_libraries(vidicant_lib PRIVATE ${OpenCV_LIBS})

# 4. Executable
add_executable(vidicant_cli src/main.cpp)
target_include_directories(vidicant_cli PRIVATE include)
target_link_libraries(vidicant_cli PRIVATE vidicant_lib)

# 5. Target-Specific Settings (The Modern Way)
target_compile_options(vidicant_cli PRIVATE 
  -Wall 
  -Wextra 
  -Wpedantic 
  -Wunused 
  -Werror
)

# 6. Testing
enable_testing()
add_subdirectory(test)
