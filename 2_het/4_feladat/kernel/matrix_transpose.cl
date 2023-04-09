__kernel void matrix_transpose(__global float* matrix, __global float* output, int cols) {
  int i = get_global_id(0); 
  int j = get_global_id(1); 
  
  int indexIn = i * cols + j;
  int indexOut = j * cols + i;
  
  output[indexOut] = matrix[indexIn];
}