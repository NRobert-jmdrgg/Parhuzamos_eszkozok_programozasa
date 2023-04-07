#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

#define ARRAY_SIZE 10

void generateArray(int *arr, size_t size, int seed);

int main() {
  cl_int err;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;
  cl_mem memobj;
  cl_program program;
  cl_kernel kernel;
  size_t global_work_size[1];
  cl_int *output;
  int i;

  int a[10];

  generateArray(&a, 10, 12345);

  for (size_t i = 0; i < 10; i++) {
    printf("%d ", a[i]);
  }

  // Create the OpenCL context and command queue
  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
    return 0;
  }

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
    return 0;
  }
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clCreateContext. Error code: %d\n", err);
    return 0;
  }
  command_queue = clCreateCommandQueue(context, device_id, 0, &err);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clCreateCommandQueue. Error code: %d\n", err);
    return 0;
  }
  // Allocate memory on the device for the output array
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                          ARRAY_SIZE * sizeof(cl_int), NULL, &err);

  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clCreateBuffer. Error code: %d\n", err);
    return 0;
  }

  // Create the kernel
  const char *source =
      "__kernel void generate_random(__global int* input) {\n"
      "    printf(\"%d \", );\n"
      "    srand(seed + gid);\n"
      "    output[gid] = rand();\n"
      "}\n";

  program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clCreateProgramWithSource. Error code: %d\n",
           err);
    return 0;
  }
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
    return 0;
  }

  kernel = clCreateKernel(program, "generate_random", &err);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clCreateKernel. Error code: %d\n", err);
    return 0;
  }

  // Set the arguments for the kernel
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clSetKernelArg. Error code: %d\n", err);
    return 0;
  }
  int seed = time(NULL);
  err = clSetKernelArg(kernel, 1, sizeof(int), &seed);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clSetKernelArg. Error code: %d\n", err);
    return 0;
  }

  // Execute the kernel
  global_work_size[0] = ARRAY_SIZE;
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size,
                               NULL, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n",
           err);
    return 0;
  }

  // Read the output back to the host
  output = (cl_int *)malloc(ARRAY_SIZE * sizeof(cl_int));
  err = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
                            ARRAY_SIZE * sizeof(cl_int), output, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clEnqueueReadBuffer. Error code: %d\n", err);
    return 0;
  }

  // Print the output
  for (i = 0; i < ARRAY_SIZE; i++) {
    printf("%d\n", output[i]);
  }

  // Clean up
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(memobj);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  free(output);

  return 0;
}

void generateArray(int *arr, size_t size, int seed) {
  srand(seed);

  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % (10 - 1 + 1) + 1;
  }
}