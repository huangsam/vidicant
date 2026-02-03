#ifndef VIDICANT_IMAGE_HPP
#define VIDICANT_IMAGE_HPP

#include <string>
#include <utility>

namespace vidicant {

std::pair<int, int> getImageDimensions(const std::string &filename);

} // namespace vidicant

#endif // VIDICANT_IMAGE_HPP
