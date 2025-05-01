#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <angle_degrees>\n";
    return -1;
  }

  std::string inputFile = argv[1];
  float angleDegrees = std::stof(argv[2]);

  // Read color image
  cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);
  if (inputImage.empty()) {
    std::cerr << "Failed to read image: " << inputFile << "\n";
    return -1;
  }

  int width = inputImage.cols;
  int height = inputImage.rows;

  float angle = angleDegrees * CV_PI / 180.0f;
  float cosA = std::cos(angle);
  float sinA = std::sin(angle);

  // Output image (3-channel color)
  cv::Mat rotatedImage(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

  int cx = width / 2;
  int cy = height / 2;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int dx = x - cx;
      int dy = y - cy;

      int srcX = static_cast<int>(cosA * dx + sinA * dy + cx);
      int srcY = static_cast<int>(-sinA * dx + cosA * dy + cy);

      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        rotatedImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
      }
    }
  }

  std::string outputFile = "rotated_output_color.png";  // Use .bmp if you want BMP
  if (!cv::imwrite(outputFile, rotatedImage)) {
    std::cerr << "Failed to save output image\n";
    return -1;
  }

  std::cout << "Saved rotated color image to " << outputFile << "\n";
  return 0;
}
