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

  // Read image in grayscale
  cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_GRAYSCALE);
  if (inputImage.empty()) {
    std::cerr << "Failed to read image: " << inputFile << "\n";
    return -1;
  }

  int width = inputImage.cols;
  int height = inputImage.rows;

  float angle = angleDegrees * CV_PI / 180.0f;
  float cosA = std::cos(angle);
  float sinA = std::sin(angle);

  cv::Mat rotatedImage(height, width, CV_8UC1, cv::Scalar(0));

  int cx = width / 2;
  int cy = height / 2;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int dx = x - cx;
      int dy = y - cy;

      int srcX = static_cast<int>(cosA * dx + sinA * dy + cx);
      int srcY = static_cast<int>(-sinA * dx + cosA * dy + cy);

      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        rotatedImage.at<uchar>(y, x) = inputImage.at<uchar>(srcY, srcX);
      }
    }
  }

  // Save the result
  std::string outputFile = "rotated_output.png";  // or .bmp
  if (!cv::imwrite(outputFile, rotatedImage)) {
    std::cerr << "Failed to save output image\n";
    return -1;
  }

  std::cout << "Saved rotated image to " << outputFile << "\n";
  return 0;
}
