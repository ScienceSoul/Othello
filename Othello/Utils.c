//
//  Utils.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

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
    
    fprintf(stdout, "FeedforwardNT: parsing the %s parameter: %s.\n", argumentName, argument);
    
    size_t len = strlen(argument);
    if (argument[0] != '{' || argument[len-1] != '}') fatal("Othello", "imput argument for network definition should start with <{> and end with <}>.");
    
    while (argument[idx] != '}') {
        if (argument[idx] == '{') {
            if (argument[idx +1] == ',' || argument[idx +1] == '{') fatal("Othello", "syntax error <{,> or <{{> in imput argument for network definition.");
            idx++;
            continue;
        }
        if (argument[idx] == ',') {
            if (argument[idx +1] == '}' || argument[idx +1] == ',') fatal("Othello", "syntax error <,}> or <,,> in imput argument for network definition.");
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
