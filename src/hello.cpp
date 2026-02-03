#include "vidicant/hello.hpp"
#include <iostream>
#include <utility>

namespace vidicant {
Hello::Hello(std::string name) : m_name(std::move(name)) {}

void Hello::greet() const {
  std::cout << "Vidicant Engine says: Hello " << m_name << "!" << std::endl;
}
} // namespace vidicant
