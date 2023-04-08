#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

char* readFromFile(const char*);

const int SAMPLE_SIZE = 1000;

/**
 * 2. Kódbetöltő készítése
 * Készítsünk egy programrészt, amelyik a kernel forráskódját egy cl
 * kiterjesztésű szöveges fájlból olvassa be! Szervezzük át az előző
 * programokat, hogy a kernelek például kernels/hello_kernel.cl útvonalról
 * legyenek betöltve!
 */
int main() {
  int i;
  cl_int err;

  char* kernel_code = readFromFile("./kernel/kernel_code.cl");

  // Get platform
  cl_uint n_platforms;
  cl_platform_id platform_id;
  err = clGetPlatformIDs(1, &platform_id, &n_platforms);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
    return 0;
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
    return 0;
  }

  // Create OpenCL context
  cl_context context =
      clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

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
    return 0;
  }

  cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

  // Create the host buffer and initialize it
  int* host_buffer = (int*)malloc(SAMPLE_SIZE * sizeof(int));
  for (i = 0; i < SAMPLE_SIZE; ++i) {
    host_buffer[i] = i;
  }

  // Create the device buffer
  cl_mem device_buffer = clCreateBuffer(
      context,
      CL_MEM_READ_WRITE,
      SAMPLE_SIZE * sizeof(int),
      NULL,
      NULL
  );

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer);
  clSetKernelArg(kernel, 1, sizeof(int), (void*)&SAMPLE_SIZE);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  clEnqueueWriteBuffer(
      command_queue,
      device_buffer,
      CL_FALSE,
      0,
      SAMPLE_SIZE * sizeof(int),
      host_buffer,
      0,
      NULL,
      NULL
  );

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
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
  clEnqueueReadBuffer(
      command_queue,
      device_buffer,
      CL_TRUE,
      0,
      SAMPLE_SIZE * sizeof(int),
      host_buffer,
      0,
      NULL,
      NULL
  );

  for (i = 0; i < SAMPLE_SIZE; ++i) {
    printf("[%d] = %d, ", i, host_buffer[i]);
  }

  // Release the resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device_id);

  free(host_buffer);
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
