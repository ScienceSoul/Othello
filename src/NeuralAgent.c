//
//  NeuralAgent.c
//  Othello
//
//  Created by Seddik hakime on 01/06/2017.
//

#include "NeuralAgent.h"

int computeAfterstatesStateValues(NeuralNetwork * _Nonnull neural, char * _Nonnull * _Nonnull board, size_t size, int * _Nonnull * _Nonnull moves, float * _Nonnull movesStateValues, int * _Nonnull * _Nonnull movesPositions, char player) {
    
    activationNode *aNodePt = NULL;
    
    // For all afterstates s′t reachable from st use NN to
    // compute V(s′t)
    int k = 0;
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            if(moves[row][col] == 0) continue;
            
            makeMove(board, row, col, player, size);
            // Create the inputs for the network
            // Values corresponding to squares are 1 when the square is taken by the learning agent
            //, -1 when it is taken by its opponent and 0 when it is empty.
            aNodePt = neural->activationsList;
            int idx = 0;
            for (int i=0; i<size; i++) {
                for (int j=0; j<size; j++) {
                    if (board[i][j] == 'O') {
                        aNodePt->a[idx] = -1.0f;
                    } else if (board[i][j] == '@') {
                        aNodePt->a[idx] = 1.0f;
                    } else aNodePt->a[idx] = 0.0f;
                    idx++;
                }
            }
            neural->feedforward((void *)neural);
            aNodePt = neural->activationsList;
            while (aNodePt != NULL && aNodePt->next != NULL) {
                aNodePt = aNodePt->next;
            }
            movesStateValues[k] = aNodePt->a[0];
            movesPositions[k][0] = row; movesPositions[k][1] = col;
            k++;
        }
    }
    
    return k;
}

void neuralAgent(NeuralNetwork * _Nonnull neural, char * _Nonnull * _Nonnull board, size_t size, int * _Nonnull * _Nonnull moves, char * _Nonnull * _Nonnull postState, int * _Nonnull ntLayers, size_t numberOfLayers, char player, float eta, float lambda, float gamma, float epsilon, bool * _Nonnull newGame, bool exploration, bool * _Nullable useTrainindData) {
    
    int min = 0;
    int bestRow = 0;                             // Best row index
    int bestCol = 0;                             // Best column index
    int inoutSizes[2];
    static bool loadedData = false;
    float movesStateValues[size*size];
    float postStateValue = 0.0f;
    float stateValue = 0.0f;
    static float reward = 0.0f;
    
    inoutSizes[0] = (int)(size*size);
    inoutSizes[1] = 1;
    
    char **tempBoard = charmatrix(0, size-1, 0, size-1);
    float **training = floatmatrix(0, 0, 0, ((size*size)+1)-1);
    int **movesPositions = intmatrix(0, (size*size)-1, 0, 1);
    
    bool alreadyTrained = false;
    if (useTrainindData != NULL) {
        alreadyTrained = (*useTrainindData == true) ? true : false;
    }
    
    if (exploration == false || alreadyTrained == true) {
        if (loadedData == false) {
            fprintf(stdout, "%s: load existing weights and biases.\n", PROGRAM_NAME);
            if (alreadyTrained == true) {
                if (loadWeightsAndBiases((void *)neural, ntLayers, numberOfLayers) != 0) {
                    fprintf(stdout, "%s: neural network training data not found, will train from scratch.\n", PROGRAM_NAME);
                }
            } else {
                if (loadWeightsAndBiases((void *)neural, ntLayers, numberOfLayers) != 0) {
                    fatal(PROGRAM_NAME, "failure reading weights and biases.");
                }
            }
            loadedData = true;
        }
    }
    
    if (*newGame == true || exploration == false) {
        // Agent first turn, we just take the move which generates
        // the highest state value
        memset(movesStateValues, 0.0f, sizeof(movesStateValues));
        memset(*movesPositions, 0, (size*2)*sizeof(int));
        int k = computeAfterstatesStateValues(neural, tempBoard, size, moves, movesStateValues, movesPositions, player);
        
        int pos = argmax(movesStateValues, k);
        bestRow = movesPositions[pos][0];
        bestCol = movesPositions[pos][1];
        
        makeMove(board, bestRow, bestCol, player, size);
        
        if (exploration == true) {
            memcpy(*postState, *board, (size*size)*sizeof(char));
            
            int agentScore, opponentScore;
            agentScore = opponentScore = 0;
            agentScore = getScore(board, '@', size);
            opponentScore = getScore(board, 'O', size);
            if (agentScore > opponentScore) { //win
                reward = 1.0f;
            } else if (agentScore < opponentScore) { //loss
                reward = 0.0f;
            } else reward = 0.5; //draw
            fprintf(stdout, "%s: reward: %f\n", PROGRAM_NAME, reward);
            
            *newGame = false;
        }
        return;
    }
    
    memset(movesStateValues, 0.0f, sizeof(movesStateValues));
    memset(*movesPositions, 0, (size*2)*sizeof(int));
    int k = computeAfterstatesStateValues(neural, tempBoard, size, moves, movesStateValues, movesPositions, player);
    
    // Select an action leading to afterstate st
    
    // Generate a random nunmber < 1 and if it is smaller than
    // the probability of exploration epsilon, select an action randomly
    
    if (((float)rand()/(float)(RAND_MAX)) * 1.0 < epsilon) {
        int pos =  (rand() % ((k-1) + 1 - min)) + min;
        bestRow = movesPositions[pos][0];
        bestCol = movesPositions[pos][1];
        stateValue = movesStateValues[pos];
    } else {
        int pos = argmax(movesStateValues, k);
        bestRow = movesPositions[pos][0];
        bestCol = movesPositions[pos][1];
        stateValue = movesStateValues[pos];
    }
    
    // Compute the target value of the previous afterstate V_new(st−1)
    // with the TD-learning algorithm
    
    postStateValue = reward + gamma*stateValue;
    if (postStateValue > 1.0f) postStateValue = 1.0f;
    
    // Use NN to compute the current value of the previous afterstate V(st−1)
    // Adjust the NN by backpropating the error V_new(st−1 ) - V(st−1)
    
    int idx = 0;
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            if (postState[row][col] == 'O') {
                training[0][idx] = -1.0f;
            } else if (postState[row][col] == '@') {
                training[0][idx] = 1.0f;
            } else training[0][idx] = 0.0f;
            idx++;
        }
    }
    training[0][inoutSizes[0]] = postStateValue;
    neural->SDG((void *)neural, training, NULL, 1, (size*size)+1, NULL, NULL, ntLayers, numberOfLayers, inoutSizes, NULL, 1, 1, eta, lambda, false, NULL);
    
    // Make the move and save it to the previous state
    
    makeMove(board, bestRow, bestCol, player, size);
    memcpy(*postState, *board, (size*size)*sizeof(char));
    
    // Compute reward from the move
    
    int agentScore, opponentScore;
    agentScore = opponentScore = 0;
    agentScore = getScore(board, '@', size);
    opponentScore = getScore(board, 'O', size);
    if (agentScore > opponentScore) { //win
        reward = 1.0f;
    } else if (agentScore < opponentScore) { //loss
        reward = 0.0f;
    } else reward = 0.5; //draw
    fprintf(stdout, "%s: reward: %f\n", PROGRAM_NAME, reward);
    
    free_cmatrix(tempBoard, 0, size-1, 0, size-1);
    free_fmatrix(training, 0, 0, 0, ((size*size)+1)-1);
    free_imatrix(movesPositions, 0, (size*size)-1, 0, 1);
}
