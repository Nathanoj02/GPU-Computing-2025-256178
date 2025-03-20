#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    if (argc != 2) {
        printf("Use 1 argument");
        exit(EXIT_FAILURE);
    }

    char *hostname = argv[1];
    printf("Hello by %s!\n", hostname);

    exit(EXIT_SUCCESS);
}