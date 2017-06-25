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

int computeAfterstatesStateValues(NeuralNetwork * _Nonnull neural, char * _Nonnull * _Nonnull board, size_t size, int * _Nonnull * _Nonnull moves, float * _Nonnull movesStateValues, int * _Nonnull * _Nonnull movesPositions, char player);

void neuralAgent(NeuralNetwork * _Nonnull neural, char * _Nonnull * _Nonnull board, size_t size, int * _Nonnull * _Nonnull moves, char * _Nonnull * _Nonnull postState, int * _Nonnull ntLayers, size_t numberOfLayers, char player, float eta, float lambda, float gamma, float epsilon, bool * _Nonnull newGame, bool exploration, bool * _Nullable useTrainindData);


