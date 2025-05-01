#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <angle_degrees>\n";
    std::cout << "Positive angles rotate counter-clockwise. Use negative for clockwise.\n";
    return -1;
  }

  std::string inputFile = argv[1];
  float angle = std::stof(argv[2]);

  cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);
  if (inputImage.empty()) {
    std::cerr << "Error: Could not read input image\n";
    return -1;
  }

  int width = inputImage.cols;
  int height = inputImage.rows;
  cv::Point2f center(width / 2.0f, height / 2.0f);

  // Compute the rotation matrix
  cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);

  // Compute the bounding rectangle to resize the output canvas
  cv::Rect2f bbox = cv::RotatedRect(center, inputImage.size(), angle).boundingRect2f();

  // Adjust transformation matrix to keep image centered
  rotMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
  rotMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

  // Apply the affine transformation
  cv::Mat rotatedImage;
  cv::warpAffine(inputImage, rotatedImage, rotMat, bbox.size(),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  std::string outputFile = "rotated_output_warpAffine.png";
  if (!cv::imwrite(outputFile, rotatedImage)) {
    std::cerr << "Failed to write output image\n";
    return -1;
  }

  std::cout << "Saved rotated image with full canvas to " << outputFile << "\n";
  return 0;
}
