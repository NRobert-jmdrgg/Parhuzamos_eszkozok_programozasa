__kernel void error(int number) {
  printf("%d\n", number / 0);
}