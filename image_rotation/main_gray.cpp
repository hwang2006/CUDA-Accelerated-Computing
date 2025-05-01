#include <dirent.h>
#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "image_rotation.h"
#include "util.h"

static int IMAGE_WIDTH = 64;
static int IMAGE_HEIGHT = 64;
static int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
static int num_src_images = 1;
static int deg = 30;

static void print_help(const char *prog_name) {
  printf("Usage: %s [-h] [-n num_src_images] [-d degree]\n", prog_name);
  printf("Options:\n");
  printf("  -h : print this page.\n");
  printf("  -n : number of source images (default: 1)\n");
  printf("  -d : rotation degree (default: 30)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "hn:d:")) != -1) {
    switch (c) {
      case 'n': num_src_images = atoi(optarg); break;
      case 'd': deg = atoi(optarg); break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  printf("Options:\n");
  printf("  Number of source images: %d\n", num_src_images);
  printf("  Rotation degrees       : %d\n\n", deg);
  printf("Positive angles rotate counter-clockwise. Use negative for clockwise.\n");
}

int main(int argc, char *argv[]) {
  parse_opt(argc, argv);

  /* Load input images */
  printf("Initializing... ");
  fflush(stdout);
  DIR *d;
  struct dirent *dir;
  int image_width, image_height;
  float *input_images, *output_images;
  char **ofname, **ifname, **ifname_trimmed;

  ofname = (char **) malloc(sizeof(char *) * num_src_images);
  ifname = (char **) malloc(sizeof(char *) * num_src_images);
  ifname_trimmed = (char **) malloc(sizeof(char *) * num_src_images);
  input_images = (float *) malloc(sizeof(float) * num_src_images * IMAGE_SIZE);
  output_images = (float *) malloc(sizeof(float) * num_src_images * IMAGE_SIZE);

  d = opendir("./images");
  if (d) {
    int i = 0;
    while (((dir = readdir(d)) != NULL) && i < num_src_images) {
      if (strcmp(dir->d_name, ".") == 0 || strcmp(dir->d_name, "..") == 0)
        continue;
      ofname[i] = (char *) malloc(sizeof(char) * 128);
      ifname[i] = (char *) malloc(sizeof(char) * 128);
      ifname_trimmed[i] = (char *) malloc(sizeof(char) * 128);
      sprintf(ifname[i], "images/%s", dir->d_name);
      sprintf(ifname_trimmed[i], dir->d_name);
      ifname_trimmed[i][strlen(ifname_trimmed[i]) - 4] = 0;  // strip .bmp
      sprintf(ofname[i], "outputs/%s_%d.bmp", ifname_trimmed[i], deg);
      i++;
    }
    closedir(d);
  }

  for (int i = 0; i < num_src_images; i++) {
    cv::Mat image = cv::imread(ifname[i], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      printf("ERROR: Could not open or find the image %s\n", ifname[i]);
      return EXIT_FAILURE;
    }

    image_width = image.cols;
    image_height = image.rows;

    if (image_width != IMAGE_WIDTH || image_height != IMAGE_HEIGHT) {
      printf("ERROR: wrong image size! Expected %dx%d, got %dx%d\n",
             IMAGE_WIDTH, IMAGE_HEIGHT, image_width, image_height);
      return EXIT_FAILURE;
    }

    image.convertTo(image, CV_32FC1, 1.0 / 255.0);
    memcpy(&input_images[i * IMAGE_SIZE], image.ptr<float>(), sizeof(float) * IMAGE_SIZE);
  }

  float theta = (float) deg / 180.0 * M_PI;
  float sin_theta = sinf(theta);
  float cos_theta = cosf(theta);

  rotate_image_init(image_width, image_height, num_src_images);
  printf("done!\n");

  /* Rotate images */
  printf("Rotating %d image(s)... ", num_src_images);
  fflush(stdout);
  double st = get_current_time();
  rotate_image(input_images, output_images, image_width, image_height,
               sin_theta, cos_theta, num_src_images);
  double en = get_current_time();
  printf("done!\n");

  /* Print performance results */
  printf("Elapsed time: %.5f sec\n", en - st);

  /* Store Images */
  printf("Storing %d image(s) in outputs/... ", num_src_images);
  fflush(stdout);

  // Ensure the "outputs" directory exists
  struct stat sb;
  if (stat("outputs", &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    if (mkdir("outputs", 0777) != 0) {
      perror("Failed to create outputs directory");
      return EXIT_FAILURE;
    }
  }

  for (int i = 0; i < num_src_images; i++) {
    cv::Mat output_mat(image_height, image_width, CV_32FC1, &output_images[i * IMAGE_SIZE]);
    cv::Mat output_uchar;
    output_mat.convertTo(output_uchar, CV_8UC1, 255.0);
    cv::imwrite(ofname[i], output_uchar);
  }
  rotate_image_cleanup();
  printf("done!\n");

  return 0;
}
