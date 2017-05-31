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

int scoreMove(char * __nonnull * __nonnull board, char player, size_t size);
int scoreMoveWithEvaluationFunction(int row, int col, size_t size);
int getScore(char * __nonnull * __nonnull board, char player, size_t size);
