# Find GoogleTest and OpenCV
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

# Create test executable
add_executable(test_image test_image.cpp)
target_include_directories(test_image PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_image vidicant_lib GTest::gmock_main ${OpenCV_LIBS})

# Add test
add_test(NAME ImageTest COMMAND test_image)
