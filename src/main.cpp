#include "vidicant/hello.hpp"
#include "vidicant/image.hpp"
#include <iostream>

int main() {
  // Instantiate your class using the vidicant namespace
  const vidicant::Hello greeter("huangsam");

  // Call the method
  greeter.greet();

  // Get image dimensions
  auto [width, height] = vidicant::getImageDimensions("examples/sample.jpg");
  if (width == -1) {
    return -1;
  }

  std::cout << "Image width: " << width << std::endl;
  std::cout << "Image height: " << height << std::endl;

  return 0;
}
