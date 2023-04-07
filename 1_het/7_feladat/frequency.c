

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
char *readFromFile(const char *);
void print_array(int *arr, size_t size);
void print_frequency(int *freq, size_t size);
void fill_array(int *arr, size_t size);
void frequency(int *arr, int *freq, size_t size);
void unique(int *arr, bool *u, size_t size);
/**
 * 7. Elemek előfordulásának a száma
 *
 *    Egy egészeket tartalmazó tömb minden eleméhez adjuk meg, hogy az adott
 * elem hányszor fordul elő a tömbben! Vizsgáljuk meg, hogy egy tömbben
 * minden érték egyedi-e!
 */
int main() {
  size_t size = 10;
  int *arr = (int *)malloc(sizeof(int) * size);
  fill_array(arr, size);
  printf("arr\n");
  print_array(arr, size);
  int *result = (int *)calloc(sizeof(int), size);

  printf("frequency\n");
  frequency(arr, result, size);
  print_frequency(result, size);

  bool u = true;
  unique(arr, &u, size);
  printf("unique? %d\n", u);

  int arr2[] = {1, 2, 3};
  bool u2 = true;
  unique(arr2, &u2, 3);
  printf("unique? %d\n", u2);

  free(arr);
  free(result);
  return 0;
}

void frequency(int *arr, int *freq, size_t size) {
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
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                       &n_devices);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
    return;
  }

  // Create OpenCL context
  cl_context context =
      clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

  char *kernel_code = readFromFile("./kernel/frequency.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_code, NULL, NULL);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "frequency", NULL);

  // Create the device buffer
  cl_mem arr_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(int), NULL, NULL);

  cl_mem freq_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      size * sizeof(int), NULL, NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&freq_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(command_queue, arr_buffer, CL_FALSE, 0,
                       size * sizeof(int), arr, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, freq_buffer, CL_FALSE, 0,
                       size * sizeof(int), freq, 0, NULL, NULL);

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size,
                         &local_work_size, 0, NULL, NULL);

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(command_queue, freq_buffer, CL_TRUE, 0,
                      size * sizeof(int), freq, 0, NULL, NULL);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device_id);

  free(kernel_code);
}

void unique(int *arr, bool *u, size_t size) {
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
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                       &n_devices);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
    return;
  }

  // Create OpenCL context
  cl_context context =
      clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

  char *kernel_code = readFromFile("./kernel/unique.cl");

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_code, NULL, NULL);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clBuildProgram. Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);
    printf("Build log:\n%s\n", log);
    free(log);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "unique", NULL);

  // Create the device buffer
  cl_mem arr_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(int), NULL, NULL);

  cl_mem unique_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL, NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&unique_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(command_queue, arr_buffer, CL_FALSE, 0,
                       size * sizeof(int), arr, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, unique_buffer, CL_FALSE, 0, sizeof(bool),
                       u, 0, NULL, NULL);

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size,
                         &local_work_size, 0, NULL, NULL);

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(command_queue, unique_buffer, CL_TRUE, 0, sizeof(bool), u,
                      0, NULL, NULL);

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device_id);

  free(kernel_code);
}

char *readFromFile(const char *filepath) {
  FILE *file = fopen(filepath, "r");
  if (!file) {
    printf("Nem sikerult megnyitni\n");
    return NULL;
  }

  fseek(file, 0L, SEEK_END);
  size_t len = ftell(file) + 1;
  fseek(file, 0L, SEEK_SET);

  char *kernel_code = (char *)malloc(len);
  fread(kernel_code, sizeof(char), len, file);
  kernel_code[len - 1] = '\0';

  fclose(file);
  return kernel_code;
}

void print_array(int *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d\n", arr[i]);
  }
}

void print_frequency(int *freq, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d - %d\n", i, freq[i]);
  }
}

void fill_array(int *arr, size_t size) {
  srand(time(0));
  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % 10;
  }
}