

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char *readFromFile(const char *);
void print_array(int *arr, size_t size);
void fill_array(int *arr, size_t size);
void replaceMissing(int *arr, size_t size);
/**
 * 5. Hiányzó elemek pótlása
 *
 * Tegyük fel, hogy egy nemnegatív egészeket tartalmazó tömbből elszórtan
 * hiányoznak elemek. Pótoljuk ezeket a szomszédos elemek átlagával!
 *
 *   Feltételezzük, hogy a hiányzó elemek mindkét szomszéd ismert.
 *   Készítsünk függvényt, amelyik ilyen bemenetet tud előállítani!
 *
 */
int main() {
  size_t size = 10;
  int *arr = (int *)malloc(sizeof(int) * size);
  printf("before:\n");
  fill_array(arr, size);
  print_array(arr, size);

  printf("after:\n");
  replaceMissing(arr, size);
  print_array(arr, size);
  free(arr);
  return 0;
}

void replaceMissing(int *arr, size_t size) {
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

  char *kernel_code = readFromFile("./kernel/replaceMissing.cl");

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
    return 0;
  }

  cl_kernel kernel = clCreateKernel(program, "replaceMissing", NULL);

  // Create the device buffer
  cl_mem arr_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     size * sizeof(int), NULL, NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_buffer);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&size);

  // Create the command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, NULL, NULL);

  // Host buffer -> Device buffer
  // WRITE BUFFERS
  clEnqueueWriteBuffer(command_queue, arr_buffer, CL_FALSE, 0,
                       size * sizeof(int), arr, 0, NULL, NULL);

  // Size specification
  size_t local_work_size = 256;
  size_t n_work_groups = (size + local_work_size + 1) / local_work_size;
  size_t global_work_size = n_work_groups * local_work_size;

  // Apply the kernel on the range
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size,
                         &local_work_size, 0, NULL, NULL);

  // Host buffer <- Device buffer
  // READ BUFFER
  clEnqueueReadBuffer(command_queue, arr_buffer, CL_TRUE, 0, size * sizeof(int),
                      arr, 0, NULL, NULL);

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

void fill_array(int *arr, size_t size) {
  srand(time(NULL));
  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % 12 - 1;
  }
}