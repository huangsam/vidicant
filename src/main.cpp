#include "vidicant/hello.hpp"

int main() {
  // Instantiate your class using the vidicant namespace
  const vidicant::Hello greeter("huangsam");

  // Call the method
  greeter.greet();

  return 0;
}
