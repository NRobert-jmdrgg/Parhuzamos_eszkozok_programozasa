#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 10

void generateArray(int* arr, size_t size, int seed);

int main() {
  int a[10];

  generateArray(&a, 10, 12345);

  for (size_t i = 0; i < 10; i++) {
    printf("%d ", a[i]);
  }

  return 0;
}

void generateArray(int* arr, size_t size, int seed) {
  srand(seed);

  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % (10 - 1 + 1) + 1;
  }
}