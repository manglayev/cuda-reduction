#include "reduction_header.cuh"


void caller()
{
  printf("STAGE 2 CALLER CPP START\n");
  wrapper();
  printf("STAGE 2 CALLER CPP END\n");
}
