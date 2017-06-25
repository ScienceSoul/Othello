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
    char * _Nonnull * _Nonnull tempBoard;
    char * _Nonnull * _Nonnull newBoard;
    int * _Nonnull * _Nonnull tempMoves;
} pthreadMoveNode;

pthreadMoveNode * _Nonnull allocatePthreadMoveNode(void);

void makeMove(char * _Nonnull * _Nonnull board, int row, int col, char player, size_t size);
int validMoves(char * _Nonnull * _Nonnull board, int * _Nonnull * _Nonnull moves, char player, size_t size);
int bestMove(char * _Nonnull * _Nonnull board, char * _Nonnull * _Nonnull newBoard, int * _Nonnull * _Nonnull moves, char player, size_t size);
void agent(char * _Nonnull * _Nonnull board, int * _Nonnull * _Nonnull moves, int numberOfMoves, char player, size_t size, char * _Nonnull method);
void * _Nullable minimaxSearch(void * _Nonnull node);
