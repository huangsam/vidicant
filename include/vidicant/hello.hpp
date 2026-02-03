#ifndef VIDICANT_HELLO_HPP
#define VIDICANT_HELLO_HPP

#include <string>

namespace vidicant {
class Hello {
public:
  // Constructor
  explicit Hello(std::string name);

  // A method to perform the greeting
  void greet() const;

private:
  std::string m_name;
};
} // namespace vidicant

#endif
