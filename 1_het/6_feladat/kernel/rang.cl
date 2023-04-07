__kernel void rang(__global int *arr, __global int *rang, int size) {
  int id = get_global_id(0);
  if (id < size) {
    
    for (int i = 0; i < size; i++) {
      if (i != id && arr[i] < arr[id]) {
        rang[id] += 1;
      }
    }
  }
}