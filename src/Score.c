//
//  Scores.c
//  Othello
//
//  Created by Seddik hakime on 25/05/2017.
//

#include "Score.h"

short evaluation_function_8x8[8][8] =
    {{120, -20, 20, 5, 5, 20, -20, 120},
    {-20, -40, -5, -5, -5, -5, -40, -20},
    {20, -5, 15, 3, 3, 15, -5, 20},
    {5, -5, 3, 3, 3, 3, -5, 5},
    {5, -5, 3, 3, 3, 3, -5, 5},
    {20, -5, 15, 3, 3, 15, -5, 20},
    {-20, -40, -5, -5, -5, -5, -40, -20},
    {120, -20, 20, 5, 5, 20, -20, 120}};

/********************************************************************
 
    Calculates the score for the current board position for the
    player. Player counters score +1, opponent counters score -1

    The second parameter identifies the player
 
    Returns the score.

*********************************************************************/
int scoreMove(char * _Nonnull * _Nonnull board, char player, size_t size) {
    
    int score = 0;                              // Score for current position
    char opponent = player == 'O' ? '@' : 'O';  // Identify opponent
    
    // Check all board squares
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            score -= board[row][col] == opponent; // Decrement for opponent
            score += board[row][col] == player;   // Increment for player
        }
    }
    return score;
}

int scoreMoveWithEvaluationFunction(int row, int col, size_t size) {
    
    return (int)evaluation_function_8x8[row][col];
}

/********************************************************************
 
    Calculates the total score for a player
 
    Returns the score.
 
*********************************************************************/
int getScore(char * _Nonnull * _Nonnull board, char player, size_t size) {
    
    int score = 0;
    
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            score += board[row][col] == player;
        }
    }
    
    return score;
}
