//
//  Scores.h
//  Othello
//
//  Created by Seddik hakime on 25/05/2017.
//

#ifndef Scores_h
#define Scores_h

#include <stdio.h>

#endif /* Scores_h */

int scoreMove(char * _Nonnull * _Nonnull board, char player, size_t size);
int scoreMoveWithEvaluationFunction(int row, int col, size_t size);
int getScore(char * _Nonnull * _Nonnull board, char player, size_t size);
