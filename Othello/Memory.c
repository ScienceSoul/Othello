//
//  memory.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include "Memory.h"

#define FI_END 1
#define FREE_ARG void*

int * __nonnull intvec(long nl, long nh)
{
    int *v;
    
    v = (int *)malloc((size_t) ((nh-nl+1+FI_END)*sizeof(int)));
    if (!v) fatal("Othello", "allocation failure for the integer vector.");
    return v-nl+FI_END;
}

float * __nonnull floatvec(long nl, long nh) {
    float *v;
    
    v = (float *)malloc((size_t) ((nh-nl+1+FI_END)*sizeof(float)));
    if (!v) fatal("Othello", "allocation failure for the float vector.");
    return v-nl+FI_END;
}

int * __nonnull * __nonnull intmatrix(long nrl, long nrh, long ncl, long nch)
{
    
    long i, nrow=nrh-nrl+1, ncol=nch-ncl+1;
    int **m;
    
    m = (int **) malloc((size_t) ((nrow+FI_END)*sizeof(int*)));
    if (!m) fatal("Othello", "allocation failure for the integer matrix 1.");
    m += FI_END;
    m -= nrl;
    
    m[nrl] = (int *) malloc((size_t) ((nrow*ncol+FI_END)*sizeof(int)));
    if (!m[nrl]) fatal("Othello", "allocation failure for the integer matrix 2.");
    m[nrl] += FI_END;
    m[nrl] -= ncl;
    
    for (i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    return m;
}

float * __nonnull * __nonnull floatmatrix(long nrl, long nrh, long ncl, long nch)
{
    long i, nrow=nrh-nrl+1, ncol=nch-ncl+1;
    float **m;
    
    m = (float **) malloc((size_t) ((nrow+FI_END)*sizeof(float*)));
    if (!m) fatal("Othello", "allocation failure for the float matrix 1.");
    m += FI_END;
    m -= nrl;
    
    m[nrl]=(float *) malloc((size_t) ((nrow*ncol+FI_END)*sizeof(float)));
    if (!m[nrl]) fatal("Othello", "allocation failure for the float matrix 2.");
    m[nrl] += FI_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    return m;
}

char * __nonnull * __nonnull charmatrix(long nrl, long nrh, long ncl, long nch) {
 
    long i, nrow=nrh-nrl+1, ncol=nch-ncl+1;
    char **m;
    
    m = (char **) malloc((size_t) ((nrow+FI_END)*sizeof(char*)));
    if (!m) fatal("Othello", "allocation failure for the char matrix 1.");
    m += FI_END;
    m -= nrl;
    
    m[nrl]=(char *) malloc((size_t) ((nrow*ncol+FI_END)*sizeof(char)));
    if (!m[nrl]) fatal("Othello", "allocation failure for the char matrix 2.");
    m[nrl] += FI_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    return m;
}

void free_ivector(int * __nonnull v, long nl, long nh)
{
    free((FREE_ARG) (v+nl-FI_END));
}

void free_fvector(float * __nonnull v, long nl, long nh)
{
    free((FREE_ARG) (v+nl-FI_END));
}

void free_imatrix(int * __nonnull * __nonnull m, long nrl, long nrh, long ncl, long nch)
{
    free((FREE_ARG) (m[nrl]+ncl-FI_END));
    free((FREE_ARG) (m+nrl-FI_END));
}

void free_fmatrix(float * __nonnull * __nonnull m, long nrl, long nrh, long ncl, long nch)
{
    free((FREE_ARG) (m[nrl]+ncl-FI_END));
    free((FREE_ARG) (m+nrl-FI_END));
}

void free_cmatrix(char * __nonnull * __nonnull m, long nrl, long nrh, long ncl, long nch)
{
    free((FREE_ARG) (m[nrl]+ncl-FI_END));
    free((FREE_ARG) (m+nrl-FI_END));
}