#include <cstdio>
#include "func.h"

extern int branch_counter[100];
extern int taken_counter[100];


int main(void) {
    foo(5);
    foo(0);
    foo(-50);
    foo(10);

    for (int i = 0; i < 1; ++ i) {
        printf("br%d\t%d / %d", i, taken_counter[i], branch_counter[i]);
    }
    printf("\n");
    
    return 0;
}