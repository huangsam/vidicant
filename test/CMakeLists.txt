# Find GoogleTest and OpenCV
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

# Create test executables
add_executable(test_image test_image.cpp)
target_include_directories(test_image PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_image vidicant_lib GTest::gmock_main ${OpenCV_LIBS})

add_executable(test_video test_video.cpp)
target_include_directories(test_video PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_video vidicant_lib GTest::gmock_main ${OpenCV_LIBS})

# Add tests
add_test(NAME ImageTest COMMAND test_image WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME VideoTest COMMAND test_video WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
