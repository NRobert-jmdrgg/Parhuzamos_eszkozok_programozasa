

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

void swap_even(cl_context context, cl_device_id device_id, char *path);
void reverse(cl_context context, cl_device_id device_id, char *path);
char *readFromFile(const char *);
void print_array(int *arr, size_t size);
/**
 * 3. Leképzés (mapping) megvalósítása
 *  Az eredménytömbbe állítsuk be a globális/lokális indexet!
 *  - Adjuk meg az elemeket visszafele sorrendben!
 *  - Cseréljük meg a szomszédos, páros és páratlan indexeken lévő elemeket!
 *  - Adjunk további példákat hasonló formában megoldható problémákra!
 */
int main() {

  // Get platform
  cl_uint n_platforms;
  cl_platform_id platform_id;
  cl_int err = clGetPlatformIDs(1, &platform_id, &n_platforms);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
    return 0;
  }

  // Get device
  cl_device_id device_id;
  cl_uint n_devices;
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                       &n_devices);
  if (err != CL_SUCCESS) {
    printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
    return 0;
  }

  // Create OpenCL context
  cl_context context =
      clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

  printf("Print reverse\n");
  reverse(context, device_id, "./kernel/reverse.cl");
  printf("swap even\n");
  swap_even(context, device_id, "./kernel/swap_even.cl");

  // Release the resources
  clReleaseContext(context);
  clReleaseDevice(device_id);

  return 0;
}

void reverse(cl_context context, cl_device_id device_id, char *path) {

  char *kernel_code = readFromFile(path);
  // Build the program
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_code, NULL, NULL);

  cl_int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Build error! Code: %d\n", err);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "print_reverse", NULL);

  const size_t array_size = 10;

  // Create the host buffer and initialize it
  int *host_buffer = (int *)malloc(array_size * sizeof(int));

  for (int i = 0; i < array_size; ++i) {
    host_buffer[i] = i;
  }

  // Create the device buffer
  cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        array_size * sizeof(int), NULL, NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_buffer);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&array_size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  clEnqueueWriteBuffer(command_queue, device_buffer, CL_FALSE, 0,
                       array_size * sizeof(int), host_buffer, 0, NULL, NULL);

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (array_size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size,
                         &local_work_size, 0, NULL, NULL);

  // Host buffer <- Device buffer
  clEnqueueReadBuffer(command_queue, device_buffer, CL_TRUE, 0,
                      array_size * sizeof(int), host_buffer, 0, NULL, NULL);

  clReleaseKernel(kernel);
  clReleaseProgram(program);

  free(host_buffer);
  free(kernel_code);
}

void swap_even(cl_context context, cl_device_id device_id, char *path) {
  char *kernel_code = readFromFile(path);

  // Build the program
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernel_code, NULL, NULL);

  cl_int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Build error! Code: %d\n", err);
    return;
  }

  cl_kernel kernel = clCreateKernel(program, "swap_even", NULL);

  const size_t array_size = 10;

  // Create the host buffer and initialize it
  int *host_buffer = (int *)malloc(array_size * sizeof(int));

  for (int i = 0; i < array_size; ++i) {
    host_buffer[i] = i;
  }

  // Create the device buffer
  cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        array_size * sizeof(int), NULL, NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_buffer);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&array_size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  clEnqueueWriteBuffer(command_queue, device_buffer, CL_FALSE, 0,
                       array_size * sizeof(int), host_buffer, 0, NULL, NULL);

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (array_size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size,
                         &local_work_size, 0, NULL, NULL);

  // Host buffer <- Device buffer
  clEnqueueReadBuffer(command_queue, device_buffer, CL_TRUE, 0,
                      array_size * sizeof(int), host_buffer, 0, NULL, NULL);

  clReleaseKernel(kernel);
  clReleaseProgram(program);

  print_array(host_buffer, array_size);

  free(host_buffer);
  free(kernel_code);
}

char *readFromFile(const char *filepath) {
  FILE *file = fopen(filepath, "rf");
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