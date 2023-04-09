__kernel void frequency(__global int *arr, __global int *freq, int size) {
   int gid = get_global_id(0);
    if (gid >= size) {
      return;
    }
    
    int element = arr[gid];
    atomic_inc(&freq[element]);
}