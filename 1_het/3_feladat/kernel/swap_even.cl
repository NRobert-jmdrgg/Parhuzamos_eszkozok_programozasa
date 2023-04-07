__kernel void swap_even(__global int *arr, int size) {
  int id = get_global_id(0);
  if (id < size && id % 2 == 0) {
    int tmp = arr[id];
    arr[id] = arr[id + 1];
    arr[id + 1] = tmp;
  }
}