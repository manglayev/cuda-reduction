#include "reduction_header.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void caller()
{
  printf("STAGE 2 CALLER CPP START\n");
  wrapper();
  printf("STAGE 2 CALLER CPP END\n");
}
