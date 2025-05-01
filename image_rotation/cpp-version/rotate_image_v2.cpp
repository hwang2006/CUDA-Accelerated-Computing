#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input image> <angle degrees>\n";
    return 1;
  }

  std::string filename = argv[1];
  double angle = std::stod(argv[2]);

  // Load input image (color)
  cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image\n";
    return 1;
  }

  // Compute rotation matrix
  cv::Point2f center(src.cols / 2.0F, src.rows / 2.0F);
  cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);

  // Determine bounding rectangle
  cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();

  // Adjust transformation matrix
  rotMat.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
  rotMat.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

  // Rotate image
  cv::Mat dst;
  cv::warpAffine(src, dst, rotMat, bbox.size());

  // Save result
  cv::imwrite("rotated_output.jpg", dst);

  std::cout << "Saved rotated image as rotated_output.jpg\n";
  return 0;
}
