__kernel void unique(__global int *arr, __global bool *u, int size) {
  if (*u) {
    int id = get_global_id(0);
    if (id < size) {
      int num = arr[id];

      for (int i = 0; i < size; i++) {
        if (i != id && arr[i] == num) {
          *u = false;
        }
      }
    }
  }
}