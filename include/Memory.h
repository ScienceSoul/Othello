//
//  memory.h
//  Othello
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef memory_h
#define memory_h

#include <stddef.h>
#include <stdarg.h>

#include "Utils.h"

#endif /* Memory_h */

int * _Nonnull intvec(long nl, long nh);
float * _Nonnull floatvec(long nl, long nh);
int * _Nonnull * _Nonnull intmatrix(long nrl, long nrh, long ncl, long nch);
float * _Nonnull * _Nonnull floatmatrix(long nrl, long nrh, long ncl, long nch);
char * _Nonnull * _Nonnull charmatrix(long nrl, long nrh, long ncl, long nch);

void free_ivector(int * _Nonnull v, long nl, long nh);
void free_fvector(float * _Nonnull v, long nl, long nh);
void free_imatrix(int * _Nonnull * _Nonnull m, long nrl, long nrh, long ncl, long nch);
void free_fmatrix(float * _Nonnull * _Nonnull m, long nrl, long nrh, long ncl, long nch);
void free_cmatrix(char * _Nonnull * _Nonnull m, long nrl, long nrh, long ncl, long nch);
