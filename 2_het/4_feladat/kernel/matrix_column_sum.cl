__kernel void matrix_column_sum(__global float* matrix, __global float* sum, int cols) {
  int i = get_global_id(0); 
  
  
  for (int j = 0; j < cols; j++) {
    sum[i] += matrix[j * cols + i];
  }

  
}