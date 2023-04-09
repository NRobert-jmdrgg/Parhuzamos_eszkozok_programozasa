__kernel void frequency(__global int *arr, __global int *freq, int size) {
  int id = get_global_id(0);
  if (id < size) {
    for (int i = 0; i < size; i++) {
      if (arr[i] == id) {
        freq[id] += 1;
      }
    }
  }
}