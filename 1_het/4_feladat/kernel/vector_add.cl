__kernel void vector_add(__global float *A, __global float *B, __global float *result, int size) {
  int id = get_global_id(0);
  if (id < size) {
    result[id] = A[id] + B[id];
  }
}