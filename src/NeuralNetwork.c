//
//  NeuralNetwork.c
//  Othello
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
    #include "cblas_f77.h"
#endif

#include "NeuralNetwork.h"


static weightNode * _Nonnull allocateWeightNode(void);
static biasNode * _Nonnull allocateBiasNode(void);

static activationNode * _Nonnull allocateActivationNode(void);
static zNode * _Nonnull allocateZNode(void);

static dcdwNode * _Nonnull allocateDcdwNode(void);
static dcdbNode * _Nonnull allocateDcdbNode(void);

static weightNode * _Nonnull initWeightsList(int * _Nonnull ntLayers, size_t numberOfLayers);
static biasNode * _Nonnull initBiasesList(int * _Nonnull ntLayers, size_t numberOfLayers);

static activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, size_t numberOfLayers);
static zNode * _Nonnull initZsList(int * _Nonnull ntLayers, size_t numberOfLayers);

static dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, size_t numberOfLayers);
static dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, size_t numberOfLayers);

static pthreadBatchNode * _Nonnull allocatePthreadBatchNode(void);

static void create(void * _Nonnull self, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nullable miniBatchSize, bool pthread);
static void destroy(void * _Nonnull self, int * _Nullable miniBatchSize, bool pthread);

static void SDG(void * _Nonnull self, float * _Nonnull * _Nonnull trainingData, float * _Nullable * _Nullable testData, size_t tr1, size_t tr2, size_t * _Nullable ts1, size_t * _Nullable ts2, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nonnull inoutSizes, int * _Nullable classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread, bool * _Nullable showTotalCost);

static void updateMiniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch, int miniBatchSize, int * _Nonnull ntLayers, size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * _Nullable pthread);

static void updateWeightsBiases(void * _Nonnull self, int miniBatchSize, size_t tr1, float eta, float lambda);

static void accumulateFromThreads(void * _Nonnull self, int miniBatchSize, bool pthread);

static void * _Nullable backpropagation(void * _Nonnull node);

static int evaluate(void * _Nonnull self, float * _Nonnull * _Nonnull testData, size_t ts1, int * _Nonnull inoutSizes);

static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, int * _Nonnull inoutSizes, int * _Nullable classifications, float lambda, bool convert);

static void __attribute__((overloadable)) feedforward(void * _Nonnull self);
static void __attribute__((overloadable)) feedforward(pthreadBatchNode * _Nonnull node);

weightNode * _Nonnull allocateWeightNode(void) {
    
    weightNode *list = (weightNode *)malloc(sizeof(weightNode));
    *list = (weightNode){.m=0, .n=0, .w=NULL, .next=NULL, .previous=NULL};
    return list;
}

biasNode * _Nonnull allocateBiasNode(void) {
    biasNode *list = (biasNode *)malloc(sizeof(biasNode));
    *list = (biasNode){.n=0, .b=NULL, .next=NULL, .previous=NULL};
    return list;
}

activationNode * _Nonnull allocateActivationNode(void) {
    activationNode *list = (activationNode *)malloc(sizeof(activationNode));
    *list = (activationNode){.n=0, .a=NULL, .next=NULL, .previous=NULL};
    return list;
}

zNode * _Nonnull allocateZNode(void) {
    zNode *list = (zNode *)malloc(sizeof(zNode));
    *list = (zNode){.n=0, .z=NULL, .next=NULL, .previous=NULL};
    return list;
}

dcdwNode * _Nonnull allocateDcdwNode(void) {
    dcdwNode *list = (dcdwNode *)malloc(sizeof(dcdwNode));
    *list = (dcdwNode){.m=0, .n=0, .dcdw=NULL, .next=NULL, .previous=NULL};
    return list;
}

dcdbNode * _Nonnull allocateDcdbNode(void) {
    dcdbNode *list = (dcdbNode *)malloc(sizeof(dcdbNode));
    *list = (dcdbNode){.n=0, .dcdb=NULL, .next=NULL, .previous=NULL};
    return list;
}

//
//  Allocate a single node in the batch
//
pthreadBatchNode * _Nonnull allocatePthreadBatchNode(void) {
    
    pthreadBatchNode *node = (pthreadBatchNode *)malloc(sizeof(pthreadBatchNode));
    *node = (pthreadBatchNode){.index=0, .max=0, .batch=NULL, .weightsList=NULL, .biasesList=NULL, .activationsList=NULL, .zsList=NULL,
        .dcdwsList=NULL, .dcdbsList=NULL, .inoutSizes=NULL};
    return node;
}

//
//  Create the weights list according to the number of layers in the network.
//  The weights are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights connecting to the same neuron.
//
//  Return a pointer to the list head.
//
weightNode * _Nonnull initWeightsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    weightNode *weightsList = allocateWeightNode();
    
    // The first weight node (i.e., layer)
    weightsList->w = floatmatrix(0, ntLayers[1]-1, 0, ntLayers[0]-1);
    weightsList->m = ntLayers[1];
    weightsList->n = ntLayers[0];
    // The rest of the weight nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    weightNode *wNodePt = weightsList;
    while (k < numberOfLayers-1) {
        weightNode *newNode = allocateWeightNode();
        newNode->w = floatmatrix(0, ntLayers[idx+1]-1, 0, ntLayers[idx]-1);
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->previous = wNodePt;
        wNodePt->next = newNode;
        wNodePt = newNode;
        k++;
        idx++;
    }
    
    wNodePt = weightsList;
    while (wNodePt != NULL) {
        for (int i = 0; i<wNodePt->m; i++) {
            for (int j=0; j<wNodePt->n; j++) {
                wNodePt->w[i][j] = randn(0.0f, 1.0f) / sqrtf((float)wNodePt->n);
            }
        }
        wNodePt = wNodePt->next;
    }
    
    return weightsList;
}

//
//  Create the biases list according to the number of layers in the network.
//  The biases are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1.
//
//  Return a pointer to the list head.
//
biasNode * _Nonnull initBiasesList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    biasNode *biasesList = allocateBiasNode();
    
    // The first bias node (i.e., layer)
    biasesList->b = floatvec(0, ntLayers[1]-1);
    biasesList->n = ntLayers[1];
    // The rest of the bias nodes (i.e., layers)
    int idx = 2;
    int k = 1;
    biasNode *bNodePt = biasesList;
    while (k < numberOfLayers-1) {
        biasNode *newNode = allocateBiasNode();
        newNode->b = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = bNodePt;
        bNodePt->next = newNode;
        bNodePt = newNode;
        k++;
        idx++;
    }
    
    bNodePt = biasesList;
    while (bNodePt != NULL) {
        for (int i = 0; i<bNodePt->n; i++) {
            bNodePt->b[i] = randn(0.0f, 1.0f);
        }
        bNodePt = bNodePt->next;
    }
    
    return biasesList;
}

//
//  Create the activations list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    activationNode *activationsList = allocateActivationNode();
    
    // The first activation node (i.e., layer)
    activationsList->a = floatvec(0, ntLayers[0]-1);
    activationsList->n = ntLayers[0];
    // The rest of the activation nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    activationNode *aNodePt = activationsList;
    while (k <= numberOfLayers-1) {
        activationNode *newNode = allocateActivationNode();
        newNode->a = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = aNodePt;
        aNodePt->next = newNode;
        aNodePt = newNode;
        k++;
        idx++;
    }
    
    aNodePt = activationsList;
    while (aNodePt != NULL) {
        memset(aNodePt->a, 0.0f, aNodePt->n*sizeof(float));
        aNodePt = aNodePt->next;
    }
    
    return activationsList;
}

//
//  Create the zs list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
zNode * _Nonnull initZsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    zNode *zsList = allocateZNode();
    
    // The first z node (i.e., layer)
    zsList->z = floatvec(0, ntLayers[0]-1);
    zsList->n = ntLayers[0];
    // The rest of the z nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    zNode *zNodePt = zsList;
    while (k <= numberOfLayers-1) {
        zNode *newNode = allocateZNode();
        newNode->z = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = zNodePt;
        zNodePt->next = newNode;
        zNodePt = newNode;
        k++;
        idx++;
    }
    
    zNodePt = zsList;
    while (zNodePt != NULL) {
        memset(zNodePt->z, 0.0f, zNodePt->n*sizeof(float));
        zNodePt = zNodePt->next;
    }
    
    return zsList;
}

//
//  Create the dC/dw list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    dcdwNode *dcdwList = allocateDcdwNode();
    
    // The first weight node (i.e., layer)
    dcdwList->dcdw = floatmatrix(0, ntLayers[1]-1, 0, ntLayers[0]-1);
    dcdwList->m = ntLayers[1];
    dcdwList->n = ntLayers[0];
    // The rest of the weight nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    dcdwNode *dcdwNodePt = dcdwList;
    while (k < numberOfLayers-1) {
        dcdwNode *newNode = allocateDcdwNode();
        newNode->dcdw = floatmatrix(0, ntLayers[idx+1]-1, 0, ntLayers[idx]-1);
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->previous = dcdwNodePt;
        dcdwNodePt->next = newNode;
        dcdwNodePt = newNode;
        k++;
        idx++;
    }
    
    return dcdwList;
}

//
//  Create the dC/db list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    dcdbNode *dcdbList = allocateDcdbNode();
    
    // The first bias node (i.e., layer)
    dcdbList->dcdb = floatvec(0, ntLayers[1]-1);
    dcdbList->n = ntLayers[1];
    // The rest of the bias nodes (i.e., layers)
    int idx = 2;
    int k = 1;
    dcdbNode *dcdbNodePt = dcdbList;
    while (k < numberOfLayers-1) {
        dcdbNode *newNode = allocateDcdbNode();
        newNode->dcdb = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = dcdbNodePt;
        dcdbNodePt->next = newNode;
        dcdbNodePt = newNode;
        k++;
        idx++;
    }
    
    return dcdbList;
}

//
// Allocate memory for a neural network
//
NeuralNetwork * _Nonnull allocateNeuralNetwork(void) {
    
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    *nn = (NeuralNetwork){.weightsList=NULL, .biasesList=NULL, .activationsList=NULL, .zsList=NULL,
        .dcdwsList=NULL, .dcdbsList=NULL, .threadDataPt=NULL, .threadTID=NULL};
    nn->create = create;
    nn->destroy = destroy;
    nn->SDG = SDG;
    nn->updateMiniBatch = updateMiniBatch;
    nn->updateWeightsBiases = updateWeightsBiases;
    nn->accumulateFromThreads = accumulateFromThreads;
    nn->backpropagation = backpropagation;
    nn->evaluate = evaluate;
    nn->totalCost = totalCost;
    nn->feedforward = feedforward;
    
    return nn;
}

//
//  Create the network layers, i.e. allocates memory for the weight, bias, activation, z, dC/dx and dC/db data structures
//
void create(void * _Nonnull self, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nullable miniBatchSize, bool pthread) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->weightsList = initWeightsList(ntLayers, numberOfLayers);
    nn->biasesList = initBiasesList(ntLayers, numberOfLayers);
    nn->activationsList = initActivationsList(ntLayers, numberOfLayers);
    nn->zsList = initZsList(ntLayers, numberOfLayers);
    nn->dcdwsList = initDcdwList(ntLayers, numberOfLayers);
    nn->dcdbsList = initDcdbList(ntLayers, numberOfLayers);
    
    if (pthread) {
        if (miniBatchSize == NULL) {
            fatal(PROGRAM_NAME, "mini batch size is NULL in network creation.");
        }
        nn->threadDataPt = (pthreadBatchNode **)malloc(*miniBatchSize * sizeof(pthreadBatchNode *));
        nn->threadTID = (pthread_t *)malloc(*miniBatchSize * sizeof(pthread_t));
        
        for (int i=0; i<*miniBatchSize; i++) {
            pthreadBatchNode *node = allocatePthreadBatchNode();
            node->max = max_array(ntLayers, numberOfLayers);
            node->weightsList = nn->weightsList;
            node->biasesList = nn->biasesList;
            node->activationsList = initActivationsList(ntLayers, numberOfLayers);
            node->zsList = initZsList(ntLayers, numberOfLayers);
            node->dcdwsList = initDcdwList(ntLayers, numberOfLayers);
            node->dcdbsList = initDcdbList(ntLayers, numberOfLayers);
            nn->threadDataPt[i] = node;
        }
    } else {
        nn->threadDataPt = (pthreadBatchNode **)malloc(1*sizeof(pthreadBatchNode *));
        pthreadBatchNode *node = allocatePthreadBatchNode();
        node->max = max_array(ntLayers, numberOfLayers);
        node->weightsList = nn->weightsList;
        node->biasesList = nn->biasesList;
        node->activationsList = nn->activationsList;
        node->zsList = nn->zsList;;
        node->dcdwsList = initDcdwList(ntLayers, numberOfLayers);
        node->dcdbsList = initDcdbList(ntLayers, numberOfLayers);
        nn->threadDataPt[0] = node;
    }
}

//
// Free-up all the memory used by a network
//
void destroy(void * _Nonnull self, int * _Nullable miniBatchSize, bool pthread) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    if (pthread) {
        if (miniBatchSize == NULL) {
            fatal(PROGRAM_NAME, "mini batch size is NULL in network destruction.");
        }
        for (int i=0; i<*miniBatchSize; i++) {
            pthreadBatchNode *node = nn->threadDataPt[i];
            node->weightsList = NULL;
            node->biasesList = NULL;
            node->inoutSizes = NULL;
            
            activationNode *aTail = node->activationsList;
            while (aTail != NULL && aTail->next != NULL) {
                aTail = aTail->next;
            }
            activationNode *aNodePt = NULL;
            while (aTail != NULL) {
                aNodePt = aTail->previous;
                free_fvector(aTail->a, 0, aTail->n);
                aTail->a = NULL;
                aTail->next = NULL;
                aTail->previous = NULL;
                free(aTail);
                aTail = aNodePt;
            }
            
            zNode *zTail = node->zsList;
            while (zTail != NULL && zTail->next != NULL) {
                zTail = zTail->next;
            }
            zNode *zNodePt = NULL;
            while (zTail != NULL) {
                zNodePt = zTail->previous;
                free_fvector(zTail->z, 0, zTail->n);
                zTail->z = NULL;
                zTail->next = NULL;
                zTail->previous = NULL;
                free(zTail);
                zTail = zNodePt;
            }
            
            dcdwNode *dcdwTail = node->dcdwsList;
            while (dcdwTail != NULL && dcdwTail->next ) {
                dcdwTail = dcdwTail->next;
            }
            dcdwNode *dcdwNodePt = NULL;
            while (dcdwTail != NULL) {
                dcdwNodePt = dcdwTail->previous;
                free_fmatrix(dcdwTail->dcdw, 0, dcdwTail->m-1, 0, dcdwTail->n-1);
                dcdwTail->dcdw = NULL;
                dcdwTail->next = NULL;
                dcdwTail->previous = NULL;
                free(dcdwTail);
                dcdwTail = dcdwNodePt;
            }
            
            dcdbNode *dcdbTail = node->dcdbsList;
            while (dcdbTail != NULL && dcdbTail->next != NULL) {
                dcdbTail = dcdbTail->next;
            }
            dcdbNode *dcdbNodePt = NULL;
            while (dcdbTail != NULL) {
                dcdbNodePt = dcdbTail->previous;
                free_fvector(dcdbTail->dcdb, 0, dcdbTail->n);
                dcdbTail->dcdb = NULL;
                dcdbTail->next = NULL;
                dcdbTail->previous = NULL;
                free(dcdbTail);
                dcdbTail = dcdbNodePt;
            }
            free(node);
        }
        free(nn->threadDataPt);
        free(nn->threadTID);
    } else {
        pthreadBatchNode *node = nn->threadDataPt[0];
        node->weightsList = NULL;
        node->biasesList = NULL;
        node->inoutSizes = NULL;
        node->activationsList = NULL;
        node->zsList = NULL;
        
        dcdwNode *dcdwTail = node->dcdwsList;
        while (dcdwTail != NULL && dcdwTail->next ) {
            dcdwTail = dcdwTail->next;
        }
        dcdwNode *dcdwNodePt = NULL;
        while (dcdwTail != NULL) {
            dcdwNodePt = dcdwTail->previous;
            free_fmatrix(dcdwTail->dcdw, 0, dcdwTail->m-1, 0, dcdwTail->n-1);
            dcdwTail->dcdw = NULL;
            dcdwTail->next = NULL;
            dcdwTail->previous = NULL;
            free(dcdwTail);
            dcdwTail = dcdwNodePt;
        }
        
        dcdbNode *dcdbTail = node->dcdbsList;
        while (dcdbTail != NULL && dcdbTail->next != NULL) {
            dcdbTail = dcdbTail->next;
        }
        dcdbNode *dcdbNodePt = NULL;
        while (dcdbTail != NULL) {
            dcdbNodePt = dcdbTail->previous;
            free_fvector(dcdbTail->dcdb, 0, dcdbTail->n);
            dcdbTail->dcdb = NULL;
            dcdbTail->next = NULL;
            dcdbTail->previous = NULL;
            free(dcdbTail);
            dcdbTail = dcdbNodePt;
        }
        free(node);
        free(nn->threadDataPt);
    }
    
    weightNode *wTail = nn->weightsList;
    while (wTail != NULL && wTail->next != NULL) {
        wTail = wTail->next;
    }
    weightNode *wNodePt = NULL;
    while (wTail != NULL) {
        wNodePt = wTail->previous;
        free_fmatrix(wTail->w, 0, wTail->m-1, 0, wTail->n-1);
        wTail->w = NULL;
        wTail->next = NULL;
        wTail->previous = NULL;
        free(wTail);
        wTail = wNodePt;
    }
    
    biasNode *bTail = nn->biasesList;
    while (bTail != NULL && bTail->next != NULL) {
        bTail = bTail->next;
    }
    biasNode *bNodePt = NULL;
    while (bTail != NULL) {
        bNodePt = bTail->previous;
        free_fvector(bTail->b, 0, bTail->n);
        bTail->b = NULL;
        bTail->next = NULL;
        bTail->previous = NULL;
        free(bTail);
        bTail = bNodePt;
    }
    
    dcdwNode *dcdwTail = nn->dcdwsList;
    while (dcdwTail != NULL && dcdwTail->next ) {
        dcdwTail = dcdwTail->next;
    }
    dcdwNode *dcdwNodePt = NULL;
    while (dcdwTail != NULL) {
        dcdwNodePt = dcdwTail->previous;
        free_fmatrix(dcdwTail->dcdw, 0, dcdwTail->m-1, 0, dcdwTail->n-1);
        dcdwTail->dcdw = NULL;
        dcdwTail->next = NULL;
        dcdwTail->previous = NULL;
        free(dcdwTail);
        dcdwTail = dcdwNodePt;
    }
    
    dcdbNode *dcdbTail = nn->dcdbsList;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    dcdbNode *dcdbNodePt = NULL;
    while (dcdbTail != NULL) {
        dcdbNodePt = dcdbTail->previous;
        free_fvector(dcdbTail->dcdb, 0, dcdbTail->n);
        dcdbTail->dcdb = NULL;
        dcdbTail->next = NULL;
        dcdbTail->previous = NULL;
        free(dcdbTail);
        dcdbTail = dcdbNodePt;
    }
    
    activationNode *aTail = nn->activationsList;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    activationNode *aNodePt = NULL;
    while (aTail != NULL) {
        aNodePt = aTail->previous;
        free_fvector(aTail->a, 0, aTail->n);
        aTail->a = NULL;
        aTail->next = NULL;
        aTail->previous = NULL;
        free(aTail);
        aTail = aNodePt;
    }
    
    zNode *zTail = nn->zsList;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    zNode *zNodePt = NULL;
    while (zTail != NULL) {
        zNodePt = zTail->previous;
        free_fvector(zTail->z, 0, zTail->n);
        zTail->z = NULL;
        zTail->next = NULL;
        zTail->previous = NULL;
        free(zTail);
        zTail = zNodePt;
    }
}

void SDG(void * _Nonnull self, float * _Nonnull * _Nonnull trainingData, float * _Nullable * _Nullable testData, size_t tr1, size_t tr2, size_t * _Nullable ts1, size_t * _Nullable ts2, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nonnull inoutSizes, int * _Nullable classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread, bool * _Nullable showTotalCost) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    if (pthread) {
        for (int i=0; i<miniBatchSize; i++) {
            pthreadBatchNode *node = nn->threadDataPt[i];
            node->inoutSizes = inoutSizes;
        }
    } else {
        pthreadBatchNode *node = nn->threadDataPt[0];
        node->inoutSizes = inoutSizes;
    }
    
    // Stochastic gradient descent
    float **miniBatch = floatmatrix(0, miniBatchSize-1, 0, tr2-1);
    int delta;
    for (int k=1; k<=epochs; k++) {
        delta = 0;
        shuffle(trainingData, tr1, tr2);
        
        fprintf(stdout, "%s: Epoch {%d/%d}:\n", PROGRAM_NAME, k, epochs);
        double rt = realtime();
        for (int l=1; l<=tr1/miniBatchSize; l++) {
            memcpy(*miniBatch, *trainingData+delta, (miniBatchSize*tr2)*sizeof(float));
            if (pthread) {
                nn->updateMiniBatch((void *)nn, miniBatch, miniBatchSize, ntLayers, numberOfLayers, tr1, eta, lambda, &pthread);
            } else {
                nn->updateMiniBatch((void *)nn, miniBatch, miniBatchSize, ntLayers, numberOfLayers, tr1, eta, lambda, NULL);
            }
            delta = delta + ((int)miniBatchSize*(int)tr2);
        }
        rt = realtime() -  rt;
        fprintf(stdout, "%s: time to complete all training data set (s): %f\n", PROGRAM_NAME, rt);
        
        if (testData != NULL) {
            fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%zu} inputs:\n", PROGRAM_NAME, k, epochs, *ts1);
            int result = nn->evaluate(self, testData, *ts1, inoutSizes);
            fprintf(stdout, "%s: Epoch {%d/%d}: {%d} / {%zu}.\n", PROGRAM_NAME, k, epochs, result, *ts1);
        }
        
        if (showTotalCost != NULL) {
            if (*showTotalCost == true) {
                float cost = nn->totalCost(self, trainingData, tr1, inoutSizes, NULL, lambda, false);
                fprintf(stdout, "%s: cost on training data: {%f}\n", PROGRAM_NAME, cost);
                
                if (testData != NULL) {
                    cost = nn->totalCost(self, testData, *ts1, inoutSizes, classifications, lambda, true);
                    fprintf(stdout, "%s: cost on test data: {%f}\n", PROGRAM_NAME, cost);
                }
            }
        }
        fprintf(stdout, "\n");
    }
    
    free_fmatrix(miniBatch, 0, miniBatchSize-1, 0, tr2-1);
}

void updateMiniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch, int miniBatchSize, int * _Nonnull ntLayers, size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * _Nullable pthread) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    bool multiThreadedBatch = false;
    if (pthread != NULL) {
        multiThreadedBatch = (*pthread == true) ? true : false;
    }
    
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    while (dcdwNodePt != NULL) {
        memset(*dcdwNodePt->dcdw, 0.0f, (dcdwNodePt->m*dcdwNodePt->n)*sizeof(float));
        dcdwNodePt = dcdwNodePt->next;
    }
    
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    while (dcdbNodePt != NULL) {
        memset(dcdbNodePt->dcdb, 0.0f, dcdbNodePt->n*sizeof(float));
        dcdbNodePt = dcdbNodePt->next;
    }
    
    double rt = realtime();
    for (int i=0; i<miniBatchSize; i++) {
        if (multiThreadedBatch) {
            pthreadBatchNode *node = nn->threadDataPt[i];
            node->index = i;
            node->batch = miniBatch;
            pthread_create(&(nn->threadTID[i]), NULL, nn->backpropagation, (void *)node);
        } else {
            pthreadBatchNode *node = nn->threadDataPt[0];
            node->index = i;
            node->batch = miniBatch;
            nn->backpropagation((void *)node);
            nn->accumulateFromThreads((void *)nn, miniBatchSize, multiThreadedBatch);
        }
    }
    
    if (multiThreadedBatch) {
        for (int i=0; i<miniBatchSize; i++) {
            pthread_join(nn->threadTID[i], NULL);
        }
        nn->accumulateFromThreads((void *)nn, miniBatchSize, multiThreadedBatch);
    }
    rt = realtime() - rt;
    fprintf(stdout, "%s: time to complete a single mini-batch (s): %f\n", PROGRAM_NAME, rt);
    
    
    nn->updateWeightsBiases((void *)nn, miniBatchSize, tr1, eta, lambda);
}

void updateWeightsBiases(void * _Nonnull self, int miniBatchSize, size_t tr1, float eta, float lambda) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Update weights
    weightNode *wNodePt = nn->weightsList;
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    while (wNodePt != NULL) {
        for (int i=0; i<wNodePt->m; i++) {
            for (int j=0; j<wNodePt->n; j++) {
                wNodePt->w[i][j] = (1.0f-eta*(lambda/tr1))*wNodePt->w[i][j] - (eta/miniBatchSize)*dcdwNodePt->dcdw[i][j];
            }
        }
        wNodePt = wNodePt->next;
        dcdwNodePt = dcdwNodePt->next;
    }
    
    // Update biases
    biasNode *bNodePt = nn->biasesList;
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    while (bNodePt != NULL) {
        for (int i=0; i<bNodePt->n; i++) {
            bNodePt->b[i] = bNodePt->b[i] - (eta/miniBatchSize)*dcdbNodePt->dcdb[i];
        }
        bNodePt = bNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
    }
}

void accumulateFromThreads(void * _Nonnull self, int miniBatchSize, bool pthread) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Accumulate dcdw and dc/db from all threads if multithreaded or
    // from a single one if serial
    
    if (pthread) {
        for (int i=0; i<miniBatchSize; i++) {
            dcdwNode *dcdwNodePt = nn->dcdwsList;
            dcdbNode *dcdbNodePt = nn->dcdbsList;
            pthreadBatchNode *node = nn->threadDataPt[i];
            dcdwNode *pthead_dcdwsPt = node->dcdwsList;
            dcdbNode *pthead_dcdbsPt = node->dcdbsList;
            while (dcdwNodePt != NULL && pthead_dcdwsPt != NULL) {
                for (int j=0; j<dcdwNodePt->m; j++) {
                    for (int k=0; k<dcdwNodePt->n; k++) {
                        dcdwNodePt->dcdw[j][k] = dcdwNodePt->dcdw[j][k] + pthead_dcdwsPt->dcdw[j][k];
                    }
                }
                for (int j=0; j<dcdbNodePt->n; j++) {
                    dcdbNodePt->dcdb[j] = dcdbNodePt->dcdb[j] + pthead_dcdbsPt->dcdb[j];
                }
                dcdwNodePt = dcdwNodePt->next;
                dcdbNodePt = dcdbNodePt->next;
                pthead_dcdwsPt = pthead_dcdwsPt->next;
                pthead_dcdbsPt = pthead_dcdbsPt->next;
            }
        }
    } else {
        dcdwNode *dcdwNodePt = nn->dcdwsList;
        dcdbNode *dcdbNodePt = nn->dcdbsList;
        pthreadBatchNode *node = nn->threadDataPt[0];
        dcdwNode *pthead_dcdwsPt = node->dcdwsList;
        dcdbNode *pthead_dcdbsPt = node->dcdbsList;
        while (dcdwNodePt != NULL && pthead_dcdwsPt != NULL) {
            for (int j=0; j<dcdwNodePt->m; j++) {
                for (int k=0; k<dcdwNodePt->n; k++) {
                    dcdwNodePt->dcdw[j][k] = dcdwNodePt->dcdw[j][k] + pthead_dcdwsPt->dcdw[j][k];
                }
            }
            for (int j=0; j<dcdbNodePt->n; j++) {
                dcdbNodePt->dcdb[j] = dcdbNodePt->dcdb[j] + pthead_dcdbsPt->dcdb[j];
            }
            dcdwNodePt = dcdwNodePt->next;
            dcdbNodePt = dcdbNodePt->next;
            pthead_dcdwsPt = pthead_dcdwsPt->next;
            pthead_dcdbsPt = pthead_dcdbsPt->next;
        }
    }
}

//
//  Return the gradient of the cross-entropy cost function C_x layers by layers
//
void * _Nullable backpropagation(void * _Nonnull node) {
    
    pthreadBatchNode *entry = (pthreadBatchNode *)node;
    
    // Activations at the input layer
    activationNode *aNodePt = entry->activationsList;
    for (int i=0; i<entry->inoutSizes[0]; i++) {
        aNodePt->a[i] = entry->batch[entry->index][i];
    }
    
    // Feedforward
    feedforward(entry);
    
    // ------------- Backward pass
    // At last layer
    
    activationNode *aTail = entry->activationsList;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    zNode *zTail = entry->zsList;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    
    float delta[entry->max];
    float buffer[entry->max];
    
    // Compute delta
    int k = entry->inoutSizes[0];
    for (int i=0; i<aTail->n; i++) {
        delta[i] = aTail->a[i] - entry->batch[entry->index][k];
        k++;
    }
    
    //dc/dw and dc/db at last layer
    dcdwNode *dcdwTail = entry->dcdwsList;
    while (dcdwTail != NULL && dcdwTail->next != NULL) {
        dcdwTail = dcdwTail->next;
    }
    dcdbNode *dcdbTail = entry->dcdbsList;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    aNodePt = aTail->previous;
    for (int i=0; i<dcdwTail->m; i++) {
        for (int j=0; j<dcdwTail->n; j++) {
            dcdwTail->dcdw[i][j] = aNodePt->a[j]*delta[i];
        }
    }
    for (int i=0; i<dcdbTail->n; i++) {
        dcdbTail->dcdb[i] = delta[i];
    }
    
    // The backward pass loop
    
    // Weights at last layer
    weightNode *wTail = entry->weightsList;
    while (wTail != NULL && wTail->next != NULL) {
        wTail = wTail->next;
    }
    
    weightNode *wNodePt = wTail;
    zNode *zNodePt = zTail->previous;
    dcdwNode *dcdwNodePt = dcdwTail->previous;
    dcdbNode *dcdbNodePt = dcdbTail->previous;
    
    while (dcdwNodePt != NULL && dcdbNodePt != NULL) {
        aNodePt = aNodePt->previous;
        
        float sp[zNodePt->n];
        for (int i=0; i<zNodePt->n; i++) {
            sp[i] = sigmoidPrime(zNodePt->z[i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)wNodePt->m, (int)wNodePt->n, 1.0, *wNodePt->w, (int)wNodePt->n, delta, 1, 0.0, buffer, 1);
        for (int i=0; i<zNodePt->n; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = aNodePt->a[j]*delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = delta[i];
        }
        
        wNodePt = wNodePt->previous;
        zNodePt = zNodePt->previous;
        dcdwNodePt = dcdwNodePt->previous;
        dcdbNodePt = dcdbNodePt->previous;
    }
    
    return NULL;
}

int evaluate(void * _Nonnull self, float * _Nonnull * _Nonnull testData, size_t ts1, int * _Nonnull inoutSizes) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float results[ts1];
    activationNode *aNodePt = NULL;
    
    int sum = 0;
    for (int k=0; k<ts1; k++) {
        aNodePt = nn->activationsList;
        for (int i=0; i<inoutSizes[0]; i++) {
            aNodePt->a[i] = testData[k][i];
        }
        nn->feedforward(self);
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        results[k] = (float)argmax(aNodePt->a, aNodePt->n);
        sum = sum + (results[k] == testData[k][inoutSizes[0]]);
    }
    
    return sum;
}

//
//  Compute the total cost function using a cross-entropy formulation
//
float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, int * _Nonnull inoutSizes, int * _Nullable classifications, float lambda, bool convert) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float norm, sum;
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<m; i++) {
        aNodePt = nn->activationsList;
        for (int j=0; j<inoutSizes[0]; j++) {
            aNodePt->a[j] = data[i][j];
        }
        nn->feedforward(self);
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        float y[aNodePt->n];
        memset(y, 0.0f, sizeof(y));
        if (convert == true) {
            for (int j=0; j<aNodePt->n; j++) {
                if (data[i][inoutSizes[0]] == classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = inoutSizes[0];
            for (int j=0; j<aNodePt->n; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(aNodePt->a, y, aNodePt->n) / m;
        
        sum = 0.0f;
        weightNode *wNodePt = nn->weightsList;
        while (wNodePt != NULL) {
            norm = frobeniusNorm(wNodePt->w, wNodePt->m, wNodePt->n);
            sum = sum + (norm*norm);
            wNodePt = wNodePt->next;
        }
        cost = cost + 0.5f*(lambda/m)*sum;
    }
    
    return cost;
}

//
//  Return the output of the network for a given activation input
//
void __attribute__((overloadable)) feedforward(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    weightNode *wNodePt = nn->weightsList;
    biasNode *bNodePt = nn->biasesList;
    activationNode *aNodePt = nn->activationsList;
    zNode *zNodePt = nn->zsList;
    
    while (wNodePt != NULL && bNodePt != NULL) {
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)wNodePt->m, (int)wNodePt->n, 1.0, *wNodePt->w, (int)wNodePt->n, aNodePt->previous->a, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, bNodePt->b, 1, zNodePt->z, 1, bNodePt->n);
#else
        for (int i=0; i<bNodePt->n; i++) {
            zNodePt->z[i] = buffer[i] + bNodePt->b[i];
        }
#endif
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = sigmoid(zNodePt->z[i]);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        wNodePt = wNodePt->next;
        bNodePt = bNodePt->next;
    }
}

void __attribute__((overloadable)) feedforward(pthreadBatchNode * _Nonnull node) {
    
    weightNode *wNodePt = node->weightsList;
    biasNode *bNodePt = node->biasesList;
    activationNode *aNodePt = node->activationsList;
    zNode *zNodePt = node->zsList;
    
    while (wNodePt != NULL && bNodePt != NULL) {
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)wNodePt->m, (int)wNodePt->n, 1.0, *wNodePt->w, (int)wNodePt->n, aNodePt->previous->a, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, bNodePt->b, 1, zNodePt->z, 1, bNodePt->n);
#else
        for (int i=0; i<bNodePt->n; i++) {
            zNodePt->z[i] = buffer[i] + bNodePt->b[i];
        }
#endif
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = sigmoid(zNodePt->z[i]);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        wNodePt = wNodePt->next;
        bNodePt = bNodePt->next;
    }
}
