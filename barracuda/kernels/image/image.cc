#include <fmt/format.h>
#include <vpx/vp8.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <cinttypes>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

namespace bcuda {

enum class ImageFormat { Webp, Jpeg, Unknown };

namespace fourcc {

constexpr inline uint32_t FourCC(char f[4]) noexcept {
  uint32_t fourcc = 0;
  fourcc |= static_cast<uint32_t>(f[0]) << 24;
  fourcc |= static_cast<uint32_t>(f[1]) << 16;
  fourcc |= static_cast<uint32_t>(f[2]) << 8;
  fourcc |= static_cast<uint32_t>(f[3]);
  return fourcc;
}

constexpr uint32_t WebP = FourCC("RIFF");
constexpr uint32_t Jpeg = FourCC("JPEG");
}  // namespace fourcc

enum class ImageType {
  Rgb888,
  Rgba8888,
};

ImageFormat GetImageFormat(uint32_t fourCC) {
  switch (fourCC) {
    case fourcc::WebP:
      return ImageFormat::Webp;
    case fourcc::Jpeg:
      return ImageFormat::Jpeg;

    default:
      return ImageFormat::Unknown;
  }
}

class NoSuchFileException : std::exception {
 public:
  explicit NoSuchFileException(std::string const& path)
      : content_{"No such file exists: "} {
    content_ += path;
  }
  char const* what() const noexcept { return content_.c_str(); }
  std::string const& content() const noexcept { return content_; }

 private:
  std::string content_;
};

Tensor<uint8_t> LoadImage(std::string const& path, ImageFormat format,
                          ImageType type = ImageType::Rgb888) {
  std::ifstream file{path.c_str(), std::ios_base::ate | std::ios_base::binary};
  BCUDA_ENSURE_TRUE_STR(type == ImageType::Rgb888,
                        "Only RGB-888 image type supported");

  if (!file.is_open()) {
    throw NoSuchFileException(path);
  }
  auto size = file.tellg();
  file.seekg(0);

  std::unique_ptr<char[]> buffer{new char[size]};

  file.read(buffer.get(), size);

  int width = -1;
  int height = -1;
  auto pixel_buffer = WebPDecodeRGB(reinterpret_cast<uint8_t*>(buffer.get()),
                                    size, &width, &height);
  BCUDA_ENSURE_TRUE_STR(pixel_buffer != nullptr, "Unable to Decode RGB image");
  BCUDA_ENSURE_TRUE(width > 0);
  BCUDA_ENSURE_TRUE(height > 0);

  auto image_tensor = Tensor<uint8_t>{
      {static_cast<size_t>(height), static_cast<size_t>(width), 3UL}};

  for (size_t i = 0; i < static_cast<size_t>(height * width * 3); i++) {
    image_tensor[i] = pixel_buffer[i];
  }

  WebPFree(pixel_buffer);

  return std::move(image_tensor);
}



}  // namespace bcuda
