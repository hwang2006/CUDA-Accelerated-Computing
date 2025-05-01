#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

#include "image_rotation.h"
#include "util.h"

static int num_src_images = 1;
static int deg = 30;
static std::string input_path = "./images";

static void print_help(const char *prog_name) {
  printf("Usage: %s [-h] [-n num_src_images] [-d degree] [-i input_path]\n", prog_name);
  printf("Options:\n");
  printf("  -h : print this page.\n");
  printf("  -n : number of source images (default: 1)\n");
  printf("  -d : rotation degree (default: 30)\n");
  printf("  -i : input image path or directory (default: ./images)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "hn:d:i:")) != -1) {
    switch (c) {
      case 'n': num_src_images = atoi(optarg); break;
      case 'd': deg = atoi(optarg); break;
      case 'i': input_path = optarg; break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  printf("Options:\n");
  printf("  Number of source images: %d\n", num_src_images);
  printf("  Rotation degrees : %d\n", deg);
  printf("  Input path : %s\n", input_path.c_str());
  printf("\n");
}

int main(int argc, char *argv[]) {
  parse_opt(argc, argv);

  std::vector<std::string> filenames;
  struct stat s;
  if (stat(input_path.c_str(), &s) == 0) {
    if (s.st_mode & S_IFDIR) {
      DIR *d = opendir(input_path.c_str());
      if (!d) {
        perror("Failed to open input directory");
        return EXIT_FAILURE;
      }
      struct dirent *dir;
      while ((dir = readdir(d)) != nullptr && filenames.size() < (size_t)num_src_images) {
        if (strcmp(dir->d_name, ".") == 0 || strcmp(dir->d_name, "..") == 0) continue;
        filenames.push_back(input_path + "/" + dir->d_name);
      }
      closedir(d);
    } else if (s.st_mode & S_IFREG) {
      filenames.push_back(input_path);
    } else {
      printf("Invalid input path: %s\n", input_path.c_str());
      return EXIT_FAILURE;
    }
  } else {
    perror("Failed to stat input path");
    return EXIT_FAILURE;
  }

  if (filenames.empty()) {
    printf("No images found in %s\n", input_path.c_str());
    return EXIT_FAILURE;
  }

  printf("Initializing...\n");

  int image_width = 0, image_height = 0;
  std::vector<cv::Mat> input_images;
  for (const auto& file : filenames) {
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
    if (img.empty()) {
      printf("Failed to read image: %s\n", file.c_str());
      return EXIT_FAILURE;
    }

    if (image_width == 0 || image_height == 0) {
      image_width = img.cols;
      image_height = img.rows;
    } else if (img.cols != image_width || img.rows != image_height) {
      printf("Resizing image %s to match reference size %dx%d\n", file.c_str(), image_width, image_height);
      cv::resize(img, img, cv::Size(image_width, image_height));
    }
    input_images.push_back(img);
  }

  float theta = (float)deg / 180.0f * M_PI;
  float sin_theta = sinf(theta);
  float cos_theta = cosf(theta);

  int image_size = image_width * image_height;
  float *input_array = new float[image_size * 3 * input_images.size()];
  float *output_array = new float[image_size * 3 * input_images.size()];

  for (size_t i = 0; i < input_images.size(); ++i) {
    cv::Mat img = input_images[i];
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
          input_array[i * image_size * 3 + c * image_size + y * image_width + x] =
            img.at<cv::Vec3b>(y, x)[c] / 255.0f;
        }
      }
    }
  }

  rotate_image_init(image_width, image_height, input_images.size());

  printf("Rotating %zu image(s)...\n", input_images.size());
  double st = get_current_time();
  rotate_image(input_array, output_array, image_width, image_height, sin_theta, cos_theta, input_images.size());
  //rotate_image_naive(input_array, output_array, image_width, image_height, sin_theta, cos_theta, input_images.size());
  //rotate_image_naive_color(input_array, output_array, image_width, image_height, sin_theta, cos_theta, input_images.size());
  double en = get_current_time();
  printf("Elapsed time: %.5f sec\n", en - st);

  printf("Storing rotated image(s) in ./outputs ...\n");

  for (size_t i = 0; i < input_images.size(); ++i) {
    cv::Mat output_img(image_height, image_width, CV_8UC3);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
          float val = output_array[i * image_size * 3 + c * image_size + y * image_width + x];
          val = std::min(std::max(val, 0.0f), 1.0f);
          output_img.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(val * 255.0f);
        }
      }
    }
    
    // Ensure the "outputs" directory exists
    struct stat sb;
    if (stat("outputs", &sb) != 0 || !S_ISDIR(sb.st_mode)) {
       if (mkdir("outputs", 0777) != 0) {
          perror("Failed to create outputs directory");
          return EXIT_FAILURE;
       }
    }

    // Construct output filename: base name + _degree + .ext
    std::string fname = filenames[i];
    size_t last_slash = fname.find_last_of("/\\");
    std::string filename_only = (last_slash == std::string::npos) ? fname : fname.substr(last_slash + 1);
    size_t dot = filename_only.find_last_of('.');
    std::string base_name = (dot == std::string::npos) ? filename_only : filename_only.substr(0, dot);
    std::string ext = (dot == std::string::npos) ? ".png" : filename_only.substr(dot);
    std::string out_name = "outputs/" + base_name + "_" + std::to_string(deg) + ext;

    cv::imwrite(out_name, output_img);
  }

  rotate_image_cleanup();
  delete[] input_array;
  delete[] output_array;

  printf("Done!\n");
  return 0;
}
