__kernel void matrix_multiplication(__global float* A, __global float* B, __global float* output, int n) {
  int i = get_global_id(0); 
  int j = get_global_id(1); 
  
  int indexOut = i * n + j;
  int sum = 0;
  for (int k = 0; k < n; k++) {
    sum += A[i * n + k] * B[k * n + j];
  }
  output[indexOut] = sum;
}