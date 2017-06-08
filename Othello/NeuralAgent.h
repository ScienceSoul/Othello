//
//  NeuralAgent.h
//  Othello
//
//  Created by Seddik hakime on 01/06/2017.
//

#ifndef NeuralAgent_h
#define NeuralAgent_h

#include <stdio.h>
#include "NeuralNetwork.h"
#include "Move.h"

#endif /* NeuralAgent_h */

int computeAfterstatesStateValues(NeuralNetwork * __nonnull neural, char * __nonnull * __nonnull board, size_t size, int * __nonnull * __nonnull moves, float * __nonnull movesStateValues, int * __nonnull * __nonnull movesPositions, char player);

void neuralAgent(NeuralNetwork * __nonnull neural, char * __nonnull * __nonnull board, size_t size, int * __nonnull * __nonnull moves, char * __nonnull * __nonnull postState, int * __nonnull ntLayers, size_t numberOfLayers, char player, float eta, float lambda, float gamma, float epsilon, bool * __nonnull newGame, bool exploration, bool * __nullable useTrainindData);


