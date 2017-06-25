//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include <stdio.h>
#include <pthread.h>
#include "Utils.h"
#include "TimeProfile.h"

#endif /* NeuralNetwork_h */

typedef struct weightNode {
    size_t m, n;
    float * _Nullable * _Nullable w;
    struct weightNode * _Nullable next;
    struct weightNode * _Nullable previous;
} weightNode;

typedef struct biasNode {
    size_t n;
    float * _Nullable b;
    struct biasNode * _Nullable next;
    struct biasNode * _Nullable previous;
} biasNode;

typedef struct activationNode {
    size_t n;
    float * _Nullable a;
    struct activationNode * _Nullable next;
    struct activationNode * _Nullable previous;
} activationNode;

typedef struct zNode {
    size_t n;
    float * _Nullable z;
    struct zNode * _Nullable next;
    struct zNode * _Nullable previous;
} zNode;

typedef struct dcdwNode {
    size_t m, n;
    float * _Nullable * _Nullable dcdw;
    struct dcdwNode * _Nullable next;
    struct dcdwNode * _Nullable previous;
} dcdwNode;

typedef struct dcdbNode {
    size_t n;
    float * _Nullable dcdb;
    struct dcdbNode * _Nullable next;
    struct dcdbNode * _Nullable previous;
} dcdbNode;

typedef struct pthreadBatchNode {
    int index;
    int max;
    float * _Nullable * _Nullable batch;
    struct weightNode * _Nullable weightsList;
    struct biasNode * _Nullable biasesList;
    struct activationNode * _Nullable activationsList;
    struct zNode * _Nullable zsList;
    struct dcdwNode * _Nullable dcdwsList;
    struct dcdbNode * _Nullable dcdbsList;
    int * _Nonnull inoutSizes;
} pthreadBatchNode;

typedef struct NeuralNetwork {
    weightNode * _Nullable weightsList;
    biasNode * _Nullable biasesList;
    activationNode * _Nullable activationsList;
    zNode * _Nullable zsList;
    dcdwNode * _Nullable dcdwsList;
    dcdbNode * _Nullable dcdbsList;
    
    pthreadBatchNode * _Nullable * _Nullable threadDataPt;
    pthread_t _Nullable * _Nullable threadTID;
    
    void (* _Nullable create)(void * _Nonnull self, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nullable miniBatchSize, bool pthread);
    void (* _Nullable destroy)(void * _Nonnull self, int * _Nullable miniBatchSize, bool pthread);
    
    void (* _Nullable SDG)(void * _Nonnull self, float * _Nonnull * _Nonnull trainingData, float * _Nullable * _Nullable testData, size_t tr1, size_t tr2, size_t * _Nullable ts1, size_t * _Nullable ts2, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nonnull inoutSizes, int * _Nullable classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread, bool * _Nullable showTotalCost);
    void (* _Nullable updateMiniBatch)(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch, int miniBatchSize, int * _Nonnull ntLayers, size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * _Nullable pthread);
    void(* _Nullable updateWeightsBiases)(void * _Nonnull self, int miniBatchSize, size_t tr1, float eta, float lambda);
    void (* _Nullable accumulateFromThreads)(void * _Nonnull self, int miniBatchSize, bool pthread);
    void * _Nullable (* _Nullable backpropagation)(void * _Nonnull node);
    void (* _Nonnull feedforward)(void * _Nonnull self);
    int (* _Nullable evaluate)(void * _Nonnull self, float * _Nonnull * _Nonnull testData, size_t ts1, int * _Nonnull inoutSizes);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, int * _Nonnull inoutSizes, int * _Nullable classifications, float lambda, bool convert);
} NeuralNetwork;

NeuralNetwork * _Nonnull allocateNeuralNetwork(void);
