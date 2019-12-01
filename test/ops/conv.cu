
int xconv_main() {
  const int filter_height = 6;
  const int filter_width = 6;
  Conv2dParams params{filter_height, filter_width, 1, 1, 3, 2};
  int seed;
  std::cout << "Enter Random Seed: ";
  std::cin >> seed;
  std::cout << std::endl;
  Tensor<float> filter_tensor = Tensor<float>::RandomUniformUnison(
      {filter_height, filter_width}, 9.6f, 10.5f, seed);

  std::cout << "Filter->" << std::string{filter_tensor.Shape()} << std::endl;
  auto filter = filter_tensor.Data();
  std::cout << "[";
  for (int j = 0; j < filter_height; j++) {
    for (int i = 0; i < filter_width; i++) {
      // filter[j * filter_width + i] = j * filter_width + i + 1;
      std::cout << filter[j * filter_width + i] << ", ";
    }
    if (j + 1 != filter_height) std::cout << std::endl;
  }

  std::cout << "]" << std::endl;

  int new_height;
  int new_width;
  auto dilated_filter = Dilate2dFilter(params, filter, &new_height, &new_width);

  std::cout << "[";
  for (int j = 0; j < new_height; j++) {
    for (int i = 0; i < new_width; i++) {
      std::cout << dilated_filter.Data()[j * new_width + i] << ", ";
    }
    if (j + 1 != new_height) std::cout << std::endl;
  }

  std::cout << "]" << std::endl;
  std::cout << "Dilated Filter->" << std::string{dilated_filter.Shape()}
            << std::endl;

  return EXIT_SUCCESS;
}