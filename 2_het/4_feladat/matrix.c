#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
void print_matrix(float* matrix, int size, int cols);
void matrix_transpose(cl_context context, cl_device_id device_id);
void matrix_multiplication(cl_context context, cl_device_id device_id);
void matrix_col_sum(cl_context context, cl_device_id device_id);
void matrix_row_sum(cl_context context, cl_device_id device_id);
char* readFromFile(const char* filepath);
void print_array(float* arr, int size);

/**
 * 4. Mátrix műveletek
 *
 * Implementáljuk a következő mátrix műveleteket OpenCL segítségével!
 *   Transzponálás
 *   Szorzás
 *   Oszlopösszeg számítás
 *   Sorösszeg számítás
 *
 * (A mátrix megadásához használjunk sor- vagy oszlopfolytonos tárolási módot.)
 */

int main() {
  cl_uint n_platforms;
  cl_platform_id platform_id;
  cl_int err = clGetPlatformIDs(1, &platform_id, &n_platforms);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
    return;
  }

  // Get device
  cl_device_id device_id;
  cl_uint n_devices;
  err = clGetDeviceIDs(
      platform_id,
      CL_DEVICE_TYPE_GPU,
      1,
      &device_id,
      &n_devices
  );
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
    return;
  }

  // Create OpenCL context
  cl_context context =
      clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

  matrix_transpose(context, device_id);
  matrix_multiplication(context, device_id);
  matrix_col_sum(context, device_id);
  matrix_row_sum(context, device_id);

  clReleaseContext(context);
  clReleaseDevice(device_id);

  return 0;
}

void matrix_transpose(cl_context context, cl_device_id device_id) {
  cl_int err;

  const int size = 16;
  const int cols = 4;
  float* matrix = (float*)malloc(sizeof(float) * size);
  float* output = (float*)malloc(sizeof(float) * size);

  for (int i = 0; i < size; i++) {
    matrix[i] = (float)i;
  }

  printf("original matrix:\n");
  print_matrix(matrix, size, cols);

  char* kernel_code = readFromFile("./kernel/matrix_transpose.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context,
      1,
      (const char**)&kernel_code,
      NULL,
      NULL
  );

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        0,
        NULL,
        &log_size
    );
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        log_size,
        log,
        NULL
    );
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "matrix_transpose", NULL);

  // Create the device buffer
  cl_mem matrix_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  cl_mem output_buffer = clCreateBuffer(
      context,
      CL_MEM_WRITE_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&matrix_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&cols);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(
      command_queue,
      matrix_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      matrix,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      output_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      output,
      0,
      NULL,
      NULL
  );

  // Size specification

  size_t global_work_size[2] = {cols, cols};

  cl_event current_event;
  // Apply the kernel on the range
  err = clEnqueueNDRangeKernel(
      command_queue,
      kernel,
      2,
      NULL,
      &global_work_size,
      NULL,
      0,
      NULL,
      &current_event
  );
  if (err != CL_SUCCESS) {
    printf(
        "[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n",
        err
    );
    return;
  }

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(
      command_queue,
      output_buffer,
      CL_TRUE,
      0,
      size * sizeof(float),
      output,
      0,
      NULL,
      NULL
  );

  clWaitForEvents(1, NULL);

  cl_ulong start;
  cl_ulong end;
  double elapsed_time;

  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start,
      NULL
  );
  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end,
      NULL
  );

  printf("transposed matrix:\n");
  print_matrix(output, size, cols);

  elapsed_time = (end - start) * 1.0e-9;
  printf("\nKernel execution time: %lf seconds\n", elapsed_time);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  free(kernel_code);
  free(matrix);
  free(output);
}

void matrix_col_sum(cl_context context, cl_device_id device_id) {
  cl_int err;

  const int size = 16;
  const int cols = 4;
  float* matrix = (float*)malloc(sizeof(float) * size);
  float* sum = (float*)calloc(sizeof(float), cols);

  for (int i = 0; i < size; i++) {
    matrix[i] = (float)i;
  }

  printf("matrix:\n");
  print_matrix(matrix, size, cols);

  char* kernel_code = readFromFile("./kernel/matrix_column_sum.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context,
      1,
      (const char**)&kernel_code,
      NULL,
      NULL
  );

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        0,
        NULL,
        &log_size
    );
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        log_size,
        log,
        NULL
    );
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "matrix_column_sum", NULL);

  // Create the device buffer
  cl_mem matrix_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  cl_mem sum_buffer = clCreateBuffer(
      context,
      CL_MEM_WRITE_ONLY,
      sizeof(float) * cols,
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&matrix_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sum_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&cols);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(
      command_queue,
      matrix_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      matrix,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      sum_buffer,
      CL_FALSE,
      0,
      cols * sizeof(float),
      sum,
      0,
      NULL,
      NULL
  );

  // Size specification

  size_t global_work_size = cols;

  cl_event current_event;
  // Apply the kernel on the range
  err = clEnqueueNDRangeKernel(
      command_queue,
      kernel,
      1,
      NULL,
      &global_work_size,
      NULL,
      0,
      NULL,
      &current_event
  );
  if (err != CL_SUCCESS) {
    printf(
        "[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n",
        err
    );
    return;
  }

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(
      command_queue,
      sum_buffer,
      CL_TRUE,
      0,
      cols * sizeof(float),
      sum,
      0,
      NULL,
      NULL
  );

  clWaitForEvents(1, NULL);

  cl_ulong start;
  cl_ulong end;
  double elapsed_time;

  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start,
      NULL
  );
  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end,
      NULL
  );

  printf("result:\n");
  print_array(sum, cols);

  elapsed_time = (end - start) * 1.0e-9;
  printf("\nKernel execution time: %lf seconds\n", elapsed_time);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  free(kernel_code);
  free(matrix);
  free(sum);
}

void matrix_row_sum(cl_context context, cl_device_id device_id) {
  cl_int err;

  const int size = 16;
  const int cols = 4;
  float* matrix = (float*)malloc(sizeof(float) * size);
  float* sum = (float*)calloc(sizeof(float), cols);

  for (int i = 0; i < size; i++) {
    matrix[i] = (float)i;
  }

  printf("matrix:\n");
  print_matrix(matrix, size, cols);

  char* kernel_code = readFromFile("./kernel/matrix_row_sum.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context,
      1,
      (const char**)&kernel_code,
      NULL,
      NULL
  );

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        0,
        NULL,
        &log_size
    );
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        log_size,
        log,
        NULL
    );
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "matrix_row_sum", NULL);

  // Create the device buffer
  cl_mem matrix_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  cl_mem sum_buffer = clCreateBuffer(
      context,
      CL_MEM_WRITE_ONLY,
      sizeof(float) * cols,
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&matrix_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sum_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&cols);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(
      command_queue,
      matrix_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      matrix,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      sum_buffer,
      CL_FALSE,
      0,
      cols * sizeof(float),
      sum,
      0,
      NULL,
      NULL
  );

  // Size specification

  size_t global_work_size = cols;

  cl_event current_event;
  // Apply the kernel on the range
  err = clEnqueueNDRangeKernel(
      command_queue,
      kernel,
      1,
      NULL,
      &global_work_size,
      NULL,
      0,
      NULL,
      &current_event
  );
  if (err != CL_SUCCESS) {
    printf(
        "[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n",
        err
    );
    return;
  }

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(
      command_queue,
      sum_buffer,
      CL_TRUE,
      0,
      cols * sizeof(float),
      sum,
      0,
      NULL,
      NULL
  );

  clWaitForEvents(1, NULL);

  cl_ulong start;
  cl_ulong end;
  double elapsed_time;

  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start,
      NULL
  );
  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end,
      NULL
  );

  printf("result:\n");
  print_array(sum, cols);

  elapsed_time = (end - start) * 1.0e-9;
  printf("\nKernel execution time: %lf seconds\n", elapsed_time);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  free(kernel_code);
  free(matrix);
  free(sum);
}

void matrix_multiplication(cl_context context, cl_device_id device_id) {
  cl_int err;

  const int size = 16;
  const int cols = 4;
  float* Amatrix = (float*)malloc(sizeof(float) * size);
  float* Bmatrix = (float*)malloc(sizeof(float) * size);
  float* output = (float*)malloc(sizeof(float) * size);

  for (int i = 0; i < size; i++) {
    Amatrix[i] = (float)i;
    Bmatrix[i] = (float)i;
  }

  printf("A matrix:\n");
  print_matrix(Amatrix, size, cols);

  printf("B matrix:\n");
  print_matrix(Bmatrix, size, cols);

  char* kernel_code = readFromFile("./kernel/matrix_multiply.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context,
      1,
      (const char**)&kernel_code,
      NULL,
      NULL
  );

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        0,
        NULL,
        &log_size
    );
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(
        program,
        device_id,
        CL_PROGRAM_BUILD_LOG,
        log_size,
        log,
        NULL
    );
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", NULL);

  // Create the device buffer
  cl_mem Amatrix_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  cl_mem Bmatrix_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );
  cl_mem output_buffer = clCreateBuffer(
      context,
      CL_MEM_WRITE_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Amatrix_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Bmatrix_buffer);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&cols);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(
      command_queue,
      Amatrix_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      Amatrix,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      Bmatrix_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      Bmatrix,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      output_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      output,
      0,
      NULL,
      NULL
  );

  // Size specification

  size_t global_work_size[2] = {cols, cols};

  cl_event current_event;
  // Apply the kernel on the range
  err = clEnqueueNDRangeKernel(
      command_queue,
      kernel,
      2,
      NULL,
      &global_work_size,
      NULL,
      0,
      NULL,
      &current_event
  );
  if (err != CL_SUCCESS) {
    printf(
        "[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n",
        err
    );
    return;
  }

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(
      command_queue,
      output_buffer,
      CL_TRUE,
      0,
      size * sizeof(float),
      output,
      0,
      NULL,
      NULL
  );

  clWaitForEvents(1, NULL);

  cl_ulong start;
  cl_ulong end;
  double elapsed_time;

  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start,
      NULL
  );
  clGetEventProfilingInfo(
      current_event,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end,
      NULL
  );

  printf("result matrix:\n");
  print_matrix(output, size, cols);

  elapsed_time = (end - start) * 1.0e-9;
  printf("\nKernel execution time: %lf seconds\n", elapsed_time);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  free(kernel_code);
  free(Amatrix);
  free(Bmatrix);
  free(output);
}

void print_matrix(float* matrix, int size, int cols) {
  for (int i = 0; i < size; i++) {
    printf("%f ", matrix[i]);
    if ((i + 1) % cols == 0) {
      printf("\n");
    }
  }
}

void print_array(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    printf("%f ", arr[i]);
  }
}

char* readFromFile(const char* filepath) {
  FILE* file = fopen(filepath, "r");
  if (!file) {
    printf("Nem sikerult megnyitni\n");
    return NULL;
  }

  fseek(file, 0L, SEEK_END);
  size_t len = ftell(file) + 1;
  fseek(file, 0L, SEEK_SET);

  char* kernel_code = (char*)malloc(len);
  fread(kernel_code, sizeof(char), len, file);
  kernel_code[len - 1] = '\0';

  fclose(file);
  return kernel_code;
}