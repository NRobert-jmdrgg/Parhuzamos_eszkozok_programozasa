__kernel void print_reverse(__global int* arr, int size) {
  int id = get_global_id(0);
  if (id < size) {
    printf("%d\n", arr[size - id - 1]);
  }
}