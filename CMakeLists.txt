cmake_minimum_required(VERSION 3.24)

project(
  vidicant
  VERSION 1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wunused -Werror")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

set(SOURCES
    src/hello.cpp
)

add_executable(vidicant_cli src/main.cpp ${SOURCES})
