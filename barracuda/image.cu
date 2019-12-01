#include "barracuda/image.h"

uint8_t* GetImage(size_t* height, size_t* width) {
  static auto image =
      bcuda::LoadImage("assets/kaido.webp", bcuda::ImageFormat::Webp);

  std::cout << std::string{image.Shape()} << std::endl;
  *height = image.Shape().At(0);
  *width = image.Shape().At(1);

  return image.Data();
}
