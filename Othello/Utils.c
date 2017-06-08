//
//  Utils.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
    #include "cblas_f77.h"
#endif

#include <dirent.h>
#include "NeuralNetwork.h"
#include "Utils.h"
#include "Memory.h"

static int formatType;
void format(char * __nullable head, char * __nullable message, int *iValue, double *dValue);

void __attribute__((overloadable)) fatal(char head[]) {
    
    formatType = 1;
    format(head, NULL, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[]) {
    
    formatType = 2;
    format(head, message, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], int n) {
    
    formatType = 3;
    format(head, message, &n, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], double n) {
    
    formatType = 4;
    format(head, message, NULL, &n);
}

void __attribute__((overloadable)) warning(char head[], char message[])
{
    fprintf(stdout, "%s: %s\n", head, message);
}

void __attribute__((overloadable)) warning(char head[], char message[], int n)
{
    fprintf(stdout, "%s: %s %d\n", head, message, n);
}

void __attribute__((overloadable)) warning(char head[], char message[], double n)
{
    fprintf(stdout, "%s: %s %f\n", head, message, n);
}

void format(char * __nullable head, char * __nullable message, int *iValue, double *dValue) {
    
    fprintf(stderr, "##                    A FATAL ERROR occured                   ##\n");
    fprintf(stderr, "##        Please look at the error log for diagnostic         ##\n");
    fprintf(stderr, "\n");
    if (formatType == 1) {
        fprintf(stderr, "%s: Program will abort...\n", head);
    } else if (formatType == 2) {
        fprintf(stderr, "%s: %s\n", head, message);
    } else if (formatType == 3) {
        fprintf(stderr, "%s: %s %d\n", head, message, *iValue);
    } else if (formatType == 4) {
        fprintf(stderr, "%s: %s %f\n", head, message, *dValue);
    }
    if (formatType == 2 || formatType == 3 || formatType == 4)
        fprintf(stderr, "Program will abort...\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "################################################################\n");
    fprintf(stderr, "################################################################\n");
    exit(-1);
}

int loadParameters(int * __nonnull ntLayers, size_t * __nonnull numberOfLayers, float * __nonnull eta, float * __nonnull lambda, float * __nonnull gamma, float * __nonnull epsilon, size_t * __nonnull numberOfGames) {
    
    // Very basic parsing of our inpute parameters file.
    // TODO: Needs to change that to something more flexible and with better input validation
    
    FILE *f1 = fopen("parameters.dat","r");
    if(!f1) {
        f1 = fopen("./params/parameters.dat","r");
        if(!f1) {
            f1 = fopen("../params/parameters.dat","r");
             if(!f1) {
                 fprintf(stdout,"FeedforwardNT: can't find the input parameters file.\n");
                 return -1;
             }
        }
    }
    
    char string[256];
    int lineCount = 1;
    do {
        fscanf(f1,"%s\n", string);
        
        if (lineCount == 1 && string[0] != '{') {
            fatal("FeedforwardNT", "syntax error in the file for the input parameters.");
        } else if (lineCount == 1) {
            lineCount++;
            continue;
        };
        
        if (string[0] == '!') continue;
        
        if (lineCount == 2) {
            parseArgument(string, "network definition", ntLayers, numberOfLayers);
        }
        if (lineCount == 3) {
            *eta = strtof(string, NULL);
        }
        if (lineCount == 4) {
            *lambda = strtof(string, NULL);
        }
        if (lineCount == 5) {
            *gamma = strtof(string, NULL);
        }
        if (lineCount == 6) {
            *epsilon = strtof(string, NULL);
        }
        if (lineCount == 7) {
            *numberOfGames = atoi(string);
        }
        lineCount++;
    } while (string[0] != '}');
    
    return 0;
}

float * __nonnull * __nonnull createTrainigData(float * __nonnull * __nonnull dataSet, size_t start, size_t end, size_t * __nonnull t1, size_t * __nonnull t2, int * __nonnull classifications, size_t numberOfClassifications, int * __nonnull inoutSizes) {
    
    int idx;
    float **trainingData = NULL;
    trainingData = floatmatrix(0, end-1, 0, (inoutSizes[0]+inoutSizes[1])-1);
    *t1 = end;
    *t2 = inoutSizes[0]+inoutSizes[1];
    
    if (inoutSizes[1] != numberOfClassifications) {
        fatal("FeedforwardNT", "the number of classifications should be equal to the number of activations.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<inoutSizes[0]; j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        idx = inoutSizes[0];
        for (int k=0; k<inoutSizes[1]; k++) {
            trainingData[i][idx] = 0.0f;
            idx++;
        }
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][inoutSizes[0]] == classifications[k]) {
                trainingData[i][inoutSizes[0]+k] = 1.0f;
            }
        }
    }
    
    return trainingData;
}

float * __nonnull * __nonnull createTestData(float * __nonnull * __nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * __nonnull t1, size_t * __nonnull t2) {
    
    float **testData = floatmatrix(0, end, 0, len2-1);
    *t1 = end;
    *t2 = len2;
    
    int idx = 0;
    for (int i=(int)start; i<start+end; i++) {
        for (int j=0; j<len2; j++) {
            testData[idx][j] = dataSet[i][j];
        }
        idx++;
    }
    return testData;
}

void shuffle(float * __nonnull * __nonnull array, size_t len1, size_t len2) {
    
    float t[len2];
    
    if (len1 > 1)
    {
        for (int i = 0; i < len1 - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (len1 - i) + 1);
            for (int k=0; k<len2; k++) {
                t[k] = array[j][k];
            }
            for (int k=0; k<len2; k++) {
                array[j][k] = array[i][k];
            }
            for (int k=0; k<len2; k++) {
                array[i][k] = t[k];
            }
        }
    }
}

void parseArgument(const char * __nonnull argument, const char * __nonnull argumentName, int * __nonnull result, size_t * __nonnull numberOfItems) {
    int idx = 0;
    *numberOfItems = 0;
    
    fprintf(stdout, "FeedforwardNT: parsing the parameter %s : %s.\n", argumentName, argument);
    
    size_t len = strlen(argument);
    if (argument[0] != '{' || argument[len-1] != '}') fatal("FeedforwardNT", "mput argument for network definition should start with <{> and end with <}>.");
    
    while (argument[idx] != '}') {
        if (argument[idx] == '{') {
            if (argument[idx +1] == ',' || argument[idx +1] == '{') fatal("FeedforwardNT", "syntax error <{,> or <{{> in imput argument for network definition.");
            idx++;
            continue;
        }
        if (argument[idx] == ',') {
            if (argument[idx +1] == '}' || argument[idx +1] == ',') fatal("FeedforwardNT", "syntax error <,}> or <,,> in imput argument for network definition.");
            (*numberOfItems)++;
            idx++;
            continue;
        } else {
            int digit = argument[idx] - '0';
            result[*numberOfItems] = result[*numberOfItems] * 10 + digit;
            idx++;
        }
    }
    (*numberOfItems)++;
}

// Generate random numbers from Normal Distribution (Gauss Distribution) with mean mu and standard deviation sigma
// using the Marsaglia and Bray method
float randn(float mu, float sigma) {
    
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }
    
    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (float) X1);
}

int __attribute__((overloadable)) min_array(int * __nonnull a, size_t num_elements) {
    
    int min = INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    
    return min;
}

int __attribute__((overloadable)) max_array(int * __nonnull a, size_t num_elements)
{
    int max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    
    return max;
}

int __attribute__((overloadable)) argmax(int * __nonnull a, size_t num_elements) {
    
    int idx=0, max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

int __attribute__((overloadable)) argmax(float * __nonnull a, size_t num_elements) {
    
    int idx=0;
    float max = -HUGE_VAL;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

//  The sigmoid fonction
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Derivative of the sigmoid function
float sigmoidPrime(float z) {
    return sigmoid(z) * (1.0f - sigmoid(z));
}

//
//  Compute the Frobenius norm of a m x n matrix
//
float frobeniusNorm(float * __nonnull * __nonnull mat, size_t m, size_t n) {
    
    float norm = 0.0f;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            norm = norm + powf(mat[i][j], 2.0f);
        }
    }
    
    return sqrtf(norm);
}

float crossEntropyCost(float * __nonnull a, float * __nonnull y, size_t n) {
    
    float cost = 0.0f;
    float buffer[n];
    
    for (int i=0; i<n; i++) {
        buffer[i] = -y[i]*logf(a[i]) - (1.0f-y[i])*logf(1.0-a[i]);
    }
    nanToNum(buffer, n);
#ifdef __APPLE__
    vDSP_sve(buffer, 1, &cost, n);
#else
    for (int i=0; i<n; i++) {
        cost = cost + buffer[i];
    }
#endif
    
    return cost;
}

void  __attribute__((overloadable)) nanToNum(float * __nonnull array, size_t n) {
    
    for (int i=0; i<n; i++) {
        if (isnan(array[i]) != 0) array[i] = 0.0f;
        
        if (isinf(array[i] != 0)) {
            if (array[i] > 0) {
                array[i] = HUGE_VALF;
            } else if (array[i] < 0) {
                array[i] = -HUGE_VALF;
            }
        }
    }
}

void storeWeightsAndBiases(void * __nonnull neural, int * __nonnull ntLayers, size_t numberOfLayers) {
    
    FILE *f1, *f2;
    DIR *dir = opendir("./training");
    if (!dir) {
        f1 = fopen("../training/weights.dat", "w");
        f2 = fopen("../training/biases.dat", "w");
    } else {
        f1 = fopen("./training/weights.dat", "w");
        f2 = fopen("./training/biases.dat", "w");
    }
    if (dir) closedir(dir);
    fprintf(f1, "{\n");
    fprintf(f2, "{\n");
    fprintf(f1, "{");
    fprintf(f2, "{");
    for (int i=0; i<numberOfLayers; i++) {
        fprintf(f1, "%d", ntLayers[i]);
        
        fprintf(f2, "%d", ntLayers[i]);
        if (i != numberOfLayers-1) {
            fprintf(f1, ",");
            fprintf(f2, ",");
        }
    }
    fprintf(f1, "}\n");
    fprintf(f2, "}\n");

    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    weightNode *wNodePt = nn->weightsList;
    biasNode *bNodePt = nn->biasesList;
    int k = 1;
    while (k <= numberOfLayers-1) {
        for (int i=0; i<ntLayers[k]; i++) {
            for (int j=0; j<ntLayers[k-1]; j++) {
                fprintf(f1, "%f\n", wNodePt->w[i][j]);
            }
            fprintf(f2, "%f\n", bNodePt->b[i]);
        }
        wNodePt = wNodePt->next;
        bNodePt = bNodePt->next;
        k++;
    }
    fprintf(f1, "}\n");
    fprintf(f2, "}\n");
    
    fclose(f1);
    fclose(f2);
}

int loadWeightsAndBiases(void * __nonnull neural, int * __nonnull ntLayers, size_t numberOfLayers) {
    
    int fileNtLayers[100];
    size_t fileNumberOfLayers;
    
    // Read weights and biases
    FILE *f1 = fopen("./training/weights.dat","r");
    if (!f1) {
        f1 = fopen("../training/weights.dat","r");
        if (!f1) {
            fprintf(stdout, "Othello: trying to load existing weights but file not found.\n");
            return -1;
        }
    }
    
    FILE *f2 = fopen("./training/biases.dat","r");
    if (!f2) {
        f2 = fopen("../training/biases.dat","r");
        if (!f2) {
            fprintf(stdout, "Othello: trying to load existing biases but file not found.\n");
            return -1;
        }
    }
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    weightNode *wNodePt = nn->weightsList;
    biasNode *bNodePt = nn->biasesList;
    
    memset(fileNtLayers, 0, sizeof(fileNtLayers));
    
    char string[1024];
    int lineCount = 1;
    int m, n;
    m = n = 0;
    int k = 1;
    int count = 0;
    while(1) {
        fscanf(f1,"%s\n", string);
        if (string[0] == '}') break;
        
        if (lineCount == 1 && string[0] != '{') {
            fatal("Othello", "syntax error in the file for the input parameters.");
        } else if (lineCount == 1) {
            lineCount++;
            continue;
        };
        
        if (lineCount == 2 && string[0] == '{') {
            parseArgument(string, "network definition in existing weights file", fileNtLayers, &fileNumberOfLayers);
            if (fileNumberOfLayers != numberOfLayers) {
                fatal("Othello", "The number of layers in the neural network from which weights are loaded is not consistent with the number of layers in the currently used network.");
            }
            for (int i=0; i<fileNumberOfLayers; i++) {
                if (ntLayers[i] != fileNtLayers[i]) {
                    fatal("Othello", "The neural network from which weights are loaded is not consistent with the one currently used by the neural agent.");
                }
            }
            lineCount++;
            continue;
        }
        wNodePt->w[m][n] = strtof(string, NULL);
        n++;
        count++;
        if (n == fileNtLayers[k-1]) {
            m++;
            n = 0;
        }
        if (count == fileNtLayers[k-1]*fileNtLayers[k]) {
            count = 0;
            k++;
            m = n = 0;
            wNodePt = wNodePt->next;
        }
        lineCount++;
    }
    
    memset(fileNtLayers, 0, sizeof(fileNtLayers));
    
    lineCount = 1;
    m = 0;
    k = 1;
    while(1) {
        fscanf(f2,"%s\n", string);
        if (string[0] == '}') break;
        
        if (lineCount == 1 && string[0] != '{') {
            fatal("Othello", "syntax error in the file for the input parameters.");
        } else if (lineCount == 1) {
            lineCount++;
            continue;
        };
        
        if (lineCount == 2 && string[0] == '{') {
            parseArgument(string, "network definition in existing biases file", fileNtLayers, &fileNumberOfLayers);
            if (fileNumberOfLayers != numberOfLayers) {
                fatal("Othello", "The number of layers in the neural network from which biases are loaded is not consistent with the number of layers in the currently used network.");
            }
            for (int i=0; i<fileNumberOfLayers; i++) {
                if (ntLayers[i] != fileNtLayers[i]) {
                    fatal("Othello", "The neural network from which biases are loaded is not consistent with the one currently used by the neural agent.");
                }
            }
            lineCount++;
            continue;
        }
        
        bNodePt->b[m] = strtof(string, NULL);
        m++;
        if (m == fileNtLayers[k]) {
            k++;
            m = 0;
            bNodePt = bNodePt->next;
        }
        lineCount++;
    }
    
    fclose(f1);
    fclose(f2);
    
    return 0;
}
