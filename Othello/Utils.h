//
//  Utils.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef Utils_h
#define Utils_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

#include "Memory.h"

#endif /* Utils_h */

void __attribute__((overloadable)) fatal(char head[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], double n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], double n);

void shuffle(float * __nonnull * __nonnull array, size_t len1, size_t len2);
void parseArgument(const char * __nonnull argument, const char * __nonnull argumentName, int * __nonnull result, size_t * __nonnull numberOfItems);
float randn(float mu, float sigma);

int __attribute__((overloadable)) min_array(int * __nonnull a, size_t num_elements);
int __attribute__((overloadable)) max_array(int * __nonnull a, size_t num_elements);

int __attribute__((overloadable)) argmax(int * __nonnull a, size_t num_elements);
int __attribute__((overloadable)) argmax(float * __nonnull a, size_t num_elements);

void  __attribute__((overloadable)) nanToNum(float * __nonnull array, size_t n);
