#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "util.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <input BMP> <angle_degrees>\n", argv[0]);
    return -1;
  }

  const char *inputFile = argv[1];
  float angleDegrees = atof(argv[2]);

  int width, height;
  float *inputImage = readImage(inputFile, &width, &height);

  float angle = angleDegrees * M_PI / 180.0f;
  float cosA = cosf(angle);
  float sinA = sinf(angle);

  float *rotatedImage = (float *) malloc(sizeof(float) * width * height);
  memset(rotatedImage, 0, sizeof(float) * width * height);

  int cx = width / 2;
  int cy = height / 2;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int dx = x - cx;
      int dy = y - cy;

      int srcX = (int)(cosA * dx + sinA * dy + cx);
      int srcY = (int)(-sinA * dx + cosA * dy + cy);

      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        rotatedImage[y * width + x] = inputImage[srcY * width + srcX];
      } else {
        rotatedImage[y * width + x] = 0;
      }
    }
  }

  storeImage(rotatedImage, "rotated_output.bmp", height, width, inputFile);

  free(inputImage);
  free(rotatedImage);

  return 0;
}

