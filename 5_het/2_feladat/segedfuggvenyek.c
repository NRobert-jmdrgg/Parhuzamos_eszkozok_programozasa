#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 10

void generateArray(int* arr, size_t size, int seed);
bool array_is_sorted(int* arr, int size);
bool array_contains_duplicates(int* arr, int size);

int main() {
  int a[10];

  generateArray(&a, 10, 12345);

  for (size_t i = 0; i < 10; i++) {
    printf("%d ", a[i]);
  }

  printf("sorted: %d\n", array_is_sorted(a, 10));
  printf("duplicate: %d\n", array_contains_duplicates(a, 10));

  return 0;
}

void generateArray(int* arr, size_t size, int seed) {
  srand(seed);

  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % (10 - 1 + 1) + 1;
  }
}

bool array_is_sorted(int* arr, int size) {
  for (int i = 1; i < size; i++) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}

bool array_contains_duplicates(int* arr, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (arr[i] == arr[j]) {
        return true;
      }
    }
  }
  return false;
}