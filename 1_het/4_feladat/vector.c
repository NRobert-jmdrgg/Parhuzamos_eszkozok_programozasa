

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

char* readFromFile(const char*);
void print_array(float* arr, size_t size);
void fill_array(float* arr, size_t size);
void vector_add(float* A, float* B, float* result, size_t size);
/**
 *
 * 4. Vektorok összeadása
 *
 *   Készítsünk programot két valós vektor összeadására!
 *   Szervezzük át a programot úgy, hogy a függvény hívásakor ne látszódjon,
 * hogy OpenCL-es implementációról van szó! Szekvenciális programmal
 * ellenőríztessük az eredmény helyességét!
 *
 */
int main() {
  size_t size = 10;
  float A[size];
  float B[size];
  float* result = (float*)calloc(sizeof(float), size);
  fill_array(A, size);
  fill_array(B, size);
  vector_add(A, B, result, size);
  print_array(result, size);
  free(result);
  return 0;
}

void vector_add(float* A, float* B, float* result, size_t size) {
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

  char* kernel_code = readFromFile("./kernel/vector_add.cl");

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
    printf("Build error! Code: %d\n", err);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  // Create the device buffer
  cl_mem a_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );
  cl_mem b_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );
  cl_mem result_buffer = clCreateBuffer(
      context,
      CL_MEM_WRITE_ONLY,
      size * sizeof(float),
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_buffer);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&result_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(
      command_queue,
      a_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      A,
      0,
      NULL,
      NULL
  );
  clEnqueueWriteBuffer(
      command_queue,
      b_buffer,
      CL_FALSE,
      0,
      size * sizeof(float),
      B,
      0,
      NULL,
      NULL
  );

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(
      command_queue,
      kernel,
      1,
      NULL,
      &global_work_size,
      &local_work_size,
      0,
      NULL,
      NULL
  );

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(
      command_queue,
      result_buffer,
      CL_TRUE,
      0,
      size * sizeof(float),
      result,
      0,
      NULL,
      NULL
  );

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device_id);

  free(kernel_code);
}

char* readFromFile(const char* filepath) {
  FILE* file = fopen(filepath, "rf");
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

void print_array(float* arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%lf\n", arr[i]);
  }
}

void fill_array(float* arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = i;
  }
}