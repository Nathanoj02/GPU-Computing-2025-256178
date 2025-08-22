#ifndef __DATA_H__
#define __DATA_H__

#include <stdlib.h>
#include <stdio.h>

#include "lsh.h"

#ifdef __cplusplus
extern "C" {
#endif

Point* read_data(size_t *num_points, const char *filename);

#ifdef __cplusplus
}
#endif

#endif // __DATA_H__