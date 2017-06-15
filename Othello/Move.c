//
//  Move.c
//  Othello
//
//  Created by Seddik hakime on 25/05/2017.
//

#include "Move.h"

//
//  Allocate a node for the possible moves
//
pthreadMoveNode * __nonnull allocatePthreadMoveNode(void) {
    
    pthreadMoveNode *node = (pthreadMoveNode *)malloc(sizeof(pthreadMoveNode));
    *node = (pthreadMoveNode){.row=0, .col=0, .score=0, .size=0, .tempBoard=NULL, .newBoard=NULL, .tempMoves=NULL};
    return node;
}

/********************************************************************
 
    Makes a move. This places the counter on a square,and reverses
    all the opponent's counters affected by the move.

    The argument player identifies the player.

*********************************************************************/
void makeMove(char * __nonnull * __nonnull board, int row, int col, char player, size_t size) {
    
    int x = 0;                                   // Row index for searching
    int y = 0;                                   // Column index for searching
    char opponent = (player == 'O')? '@' : 'O';  // Identify opponent
    
    board[row][col] = player;           /* Place the player counter   */
    
    // Check all the squares around this square
    // for the opponents counter
    for(int rowdelta = -1; rowdelta <= 1; rowdelta++) {
        for(int coldelta = -1; coldelta <= 1; coldelta++) {
            // Don't check off the board, or the current square
            if(row + rowdelta < 0 || row + rowdelta >= size ||
               col + coldelta < 0 || col + coldelta >= size ||
               (rowdelta==0 && coldelta== 0)) continue;
            
            // Now check the square
            if(board[row + rowdelta][col + coldelta] == opponent) {
                // If we find the opponent, search in the same direction
                // for a player counter
                x = row + rowdelta;        // Move to opponent
                y = col + coldelta;        // square
                
                for(;;) {
                    x += rowdelta;           // Move to the
                    y += coldelta;           // next square
                    
                    // If we are off the board give up */
                    if(x < 0 || x >= size || y < 0 || y >= size) break;
                    
                    // If the square is blank give up */
                    if(board[x][y] == ' ') break;
                    
                    // If we find the player counter, go backwards from here */
                    // changing all the opponents counters to player         */
                    if(board[x][y] == player)
                    {
                        while(board[x-=rowdelta][y-=coldelta]==opponent) /* Opponent? */
                            board[x][y] = player;    // Yes, change it
                        break;                       //  We are done
                    }
                }
            }
        }
    }
}

/***************************************************************************
 
    Calculates which squares are valid moves for player. Valid moves 
    are recorded as 1, 0 otherwise.
 
    The second parameter is the moves array.
    The third parameter identifies the player to make the move.
 
    Returns valid move count.
 
*****************************************************************************/
int validMoves(char * __nonnull * __nonnull board, int * __nonnull * __nonnull moves, char player, size_t size) {
 
    int x = 0;            // Row index when searching
    int y = 0;            // Column index when searching
    int numberOfMoves = 0;
    
    // Set the opponent
    char opponent = (player == 'O')? '@' : 'O';
    
    // Initialize moves array to zero
    memset(*moves, 0, (size*size)*sizeof(int));
    
    // Find squares for valid moves.
    // A valid move must be on a blank square and must enclose
    // at least one opponent square between two player squares
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            if(board[row][col] != ' ')     // Is it a blank square?  */
                continue;                  // No - so on to the next */
            
            // Check all the squares around the blank square
            // for the opponents counter
            for(int rowdelta = -1; rowdelta <= 1; rowdelta++) {
                for(int coldelta = -1; coldelta <= 1; coldelta++) {
                    // Don't check outside the array, or the current square
                    if(row + rowdelta < 0 || row + rowdelta >= size ||
                       col + coldelta < 0 || col + coldelta >= size ||
                       (rowdelta==0 && coldelta==0)) continue;
                    
                    // Now check the square
                    if(board[row + rowdelta][col + coldelta] == opponent) {
                        // If we find the opponent, move in the delta direction
                        // over opponent counters searching for a player counter
                        x = row + rowdelta;                // Move to
                        y = col + coldelta;                // opponent square
                        
                        // Look for a player square in the delta direction
                        for(;;) {
                            x += rowdelta;                  // Go to next square
                            y += coldelta;                  // in delta direction
                            
                            // If we move outside the array, give up
                            if(x < 0 || x >= size || y < 0 || y >= size) break;
                            
                            // If we find a blank square, give up
                            if(board[x][y] == ' ') break;
                            // If the square has a player counter
                            //  then we have a valid move
                            if(board[x][y] == player) {
                                moves[row][col] = 1;   // Mark as valid
                                numberOfMoves++;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    // FIXME:  seems to return in some occasions wrong number of moves
    return numberOfMoves;
}

/***********************************************************************
    
    Calculates the score for the best move out of the valid moves
    for the player in the current position.
 
    The second argument is the local copy of the board from the caller
    The third argument is the moves array defining valid moves.
    The fourth argument identifies the player
    
    Return the score for the best move

************************************************************************/
int bestMove(char * __nonnull * __nonnull board, char * __nonnull * __nonnull newBoard, int * __nonnull * __nonnull moves, char player, size_t size) {
    
    int score = 0;                       // Best score
    int newScore = 0;                    // Score for current move
    
    // Local copy of board

    // Check all valid moves to find the best
    for(int row = 0 ; row<size ; row++) {
        for(int col = 0 ; col<size ; col++) {
            if(!moves[row][col])             // Not a valid move?
                continue;                    // Go to the next
            
            // Copy the board
            memcpy(*newBoard, *board, (size*size)*sizeof(char));
            
            // Make move on the board copy
            makeMove(newBoard, row, col, player, size);
            
            // Get score for move
            newScore = scoreMove(newBoard, player, size);
            
            if(score < newScore)         // Is it better?
                score = newScore;        // Yes, save it as best score
        }
    }
    
    return score;
}

void * __nullable minimaxSearch(void * __nonnull node) {
    
    pthreadMoveNode *entry = (pthreadMoveNode *)node;
    
    // Make move on the temporary board
    makeMove(entry->tempBoard, entry->row, entry->col, entry->player, entry->size);
    
    // Find valid moves for the opponent after this move
    validMoves(entry->tempBoard, entry->tempMoves, entry->opponent, entry->size);
    
    // Now find the score for the opponents best move
    entry->score = bestMove(entry->tempBoard, entry->newBoard, entry->tempMoves, entry->opponent, entry->size);
    
    return NULL;
}

/************************************************************************
 
    Finds the best move for the agent:
        - Compute the move for which the opponent's best possible move 
          score is a minimum.
        - Use a simple position evaluation function to compute 
          the agent's next move

    The second argument is the moves array containing valid moves.
    The third argument identifies the computer.
    The fifth argument indicates the agent moving method:
        -minimax or -evaluation-function
 
*************************************************************************/
void agent(char * __nonnull * __nonnull board, int * __nonnull * __nonnull moves, int numberOfMoves, char player, size_t size, char * __nonnull method) {
    
    int bestRow = 0;                             // Best row index
    int bestCol = 0;                             // Best column index
    int newScore = 0;                            // Score for current move
    int score = 0;                               // Minimum opponent score
    char opponent = (player == 'O')? '@' : 'O';  // Identify opponent
    
    pthreadMoveNode **threadDataPt = NULL;
    pthread_t *threadTID = NULL;
    
    // Local copy of board and local valid moves array
    if (strcmp(method, "-minimax") == 0) {
        threadDataPt = (pthreadMoveNode **)malloc(numberOfMoves * sizeof(pthreadMoveNode *));
        threadTID = (pthread_t *)malloc(numberOfMoves * sizeof(pthread_t));
        
        for (int i=0; i<numberOfMoves; i++) {
            pthreadMoveNode *node = allocatePthreadMoveNode();
            node->tempBoard = charmatrix(0, size-1, 0, size-1);
            node->newBoard = charmatrix(0, size-1, 0, size-1);
            node->tempMoves = intmatrix(0, size-1, 0, size-1);
            
            // Copies of the current board
            memcpy(*(node->tempBoard), *board, (size*size)*sizeof(char));

            memset(*(node->newBoard), 0, (size*size)*sizeof(char));
            memset(*(node->tempMoves), 0, (size*size)*sizeof(int));
            threadDataPt[i] = node;
        }
    }
    
    if (strcmp(method, "-minimax") == 0) {
        score = 100;
    } else if (strcmp(method, "-evaluation-function") == 0) {
        score = -INT_MAX;
    }
    
    // Go through all valid moves
    int idx = 0;
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            if(moves[row][col] == 0) continue;
            
            if (strcmp(method, "-minimax") == 0) {
                
                // Compute the score of the opponent for each possible move of the agent
                // This score is computed from the best move out of the valid moves
                // for the player in the current position.

                pthreadMoveNode *node = threadDataPt[idx];
                node->row = row;
                node->col = col;
                node->player = player;
                node->opponent = opponent;
                node->size = size;
                pthread_create(&(threadTID[idx]), NULL, minimaxSearch, (void *)node);
                idx++;
            } else if (strcmp(method, "-evaluation-function") == 0) {
                if (size != 8) {
                    fatal("Othello", "valuation funtion for board smaller than 8 x 8 not supported yet.");
                }
                newScore = scoreMoveWithEvaluationFunction(row, col, size);
                if (newScore > score) {
                    score = newScore;
                    bestRow = row;
                    bestCol = col;
                }
            }
        }
    }
    
    if (strcmp(method, "-minimax") == 0) {
        for (int i=0; i<idx; i++) {
            pthread_join(threadTID[i], NULL);
        }
        for (int i=0; i<idx; i++) {
            pthreadMoveNode *node = threadDataPt[i];
            if (node->score < score) {
                score = node->score;
                bestRow = node->row;
                bestCol = node->col;
            }
        }
    }
    
    // Make the best move
    makeMove(board, bestRow, bestCol, player, size);

    if (strcmp(method, "-minimax") == 0) {
        for (int i=0; i<numberOfMoves; i++) {
            pthreadMoveNode *node = threadDataPt[i];
            free_cmatrix(node->tempBoard, 0, size-1, 0, size-1);
            free_cmatrix(node->newBoard, 0, size-1, 0, size-1);
            free_imatrix(node->tempMoves, 0, size-1, 0, size-1);
            free(node);
        }
        free(threadDataPt);
        free(threadTID);
    }
}
