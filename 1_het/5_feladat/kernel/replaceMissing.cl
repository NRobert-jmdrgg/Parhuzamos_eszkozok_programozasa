__kernel void replaceMissing(__global int* arr, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    if (arr[gid] == -1) {
      int sum = 0;
    
      if (gid > 0) {
        sum += arr[gid - 1];
      }

      if (gid < size - 1) {
        sum += arr[gid + 1];
      }

      arr[gid] = sum / 2;
    } 
  }
}
