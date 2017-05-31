//
//  Move.h
//  Othello
//
//  Created by Seddik hakime on 25/05/2017.
//

#ifndef Move_h
#define Move_h

#include <stdio.h>
#include <pthread.h>

#include "Score.h"
#include "Utils.h"

#endif /* Move_h */

typedef struct pthreadMoveNode {
    int row;
    int col;
    int score;
    char player;
    char opponent;
    size_t size;
    char * __nonnull * __nonnull tempBoard;
    char * __nonnull * __nonnull newBoard;
    int * __nonnull * __nonnull tempMoves;
} pthreadMoveNode;

pthreadMoveNode * __nonnull allocatePthreadMoveNode(void);

void makeMove(char * __nonnull * __nonnull board, int row, int col, char player, size_t size);
int validMoves(char * __nonnull * __nonnull board, int * __nonnull * __nonnull moves, char player, size_t size);
int bestMove(char * __nonnull * __nonnull board, char * __nonnull * __nonnull newBoard, int * __nonnull * __nonnull moves, char player, size_t size);
void agent(char * __nonnull * __nonnull board, int * __nonnull * __nonnull moves, int numberOfMoves, char player, size_t size, char * __nonnull method);
void * __nullable minimaxSearch(void * __nonnull node);
