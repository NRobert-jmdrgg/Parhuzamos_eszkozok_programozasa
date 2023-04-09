__kernel void matrix_row_sum(__global float* matrix, __global float* sum, int cols) {
  int j = get_global_id(0); 
  
  
  for (int i = 0; i < cols; i++) {
    sum[j] += matrix[j * cols + i];
  }

  
}