//
//  HumanAgainstAgent.c
//  Othello
//
//  Created by Seddik hakime on 31/05/2017.
//

#include "HumanAgainstAgent.h"
#include "Board.h"
#include "Move.h"

void humanAgainstAgent(char * __nonnull * __nonnull board, size_t size, int * __nonnull * __nonnull moves, char * __nonnull  method) {
    
    int noOfGames = 0;                // Number of games
    int noOfMoves = 0;                // Count of moves
    int invalidMoves = 0;             // Invalid move count
    int compScore = 0;                // Computer score
    int userScore = 0;                // Player score
    char y = 0;                       // Column letter
    int c = 0;                        // Column number
    int r = 0;                        // Row number
    char retr = 0;                    // Return key
    int player = 0;                   // Player indicator
    
    // On even games the player starts;
    // On odd games the computer starts
    player = ++noOfGames % 2;
    
    // Starts with four for the move count
    noOfMoves = 4;
    
    // Blank all the board squares */
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            board[row][col] = ' ';
        }
    }
    
    // Place the initial four counters in the center
    board[size/2 - 1][size/2 - 1] = board[size/2][size/2] = 'O';
    board[size/2 - 1][size/2]     = board[size/2][size/2 - 1] = '@';
    
    // The game play loop
    do {
        
        displayBoard(board, size);
        fprintf(stdout, "Othello: number of moves so far: %d\n", noOfMoves);
        
        if(player++ % 2) { // It is the player's turn
            if(validMoves(board, moves, 'O', size)) {
                // Read player moves until a valid move is entered
                int count = 0;
                for(;;) {
                    fflush(stdin);                                                   // Flush the keyboard buffer
                    fprintf(stdout,"Othello: please enter your move (row column): ");
                    fscanf(stdin, "%d%c", &r, &y);                                   // Read input
                    c = tolower(y) - 'a';                                            // Convert to column index
                    r--;                                                             // Convert to row index
                    if(r>=0 && c>=0 && r<size && c<size && moves[r][c]) {
                        makeMove(board, r, c, 'O', size);
                        noOfMoves++;
                        break;
                    } else {
                        fprintf(stdout,"Othello: not a valid move, try again.\n");
                    }
                    // FIXME: Temporary work around when the loop goes infinite
                    // if the input is in the wrong format, e.g., c5
                    count++;
                    if (count > 10) {
                        fatal("Othello", "something went wrong with the input.");
                    }
                }
            } else {
                // No valid moves
                if(++invalidMoves<2) {
                    fflush(stdin);
                    fprintf(stdout,"\nOthello: you have to pass, press return");
                    fscanf(stdin, "%c", &retr);
                } else {
                    fprintf(stdout,"\nOthello: neither of us can go, so the game is over.\n");
                    break;
                }
            }
        } else { // It is the agent's turn
            int numberOfMoves = validMoves(board, moves, '@', size);
            if(numberOfMoves) {
                // Reset invalid count
                invalidMoves = 0;
                agent(board, moves, numberOfMoves, '@', size, method);
                noOfMoves++;
            } else {
                if(++invalidMoves<2) {
                    fprintf(stdout,"\nOthello: I have to pass, your go\n"); // No valid move
                }
                else {
                    fprintf(stdout,"\nOthello: neither of us can go, so the game is over.\n");
                    break;
                }
            }
        }
    } while(noOfMoves < size*size && invalidMoves<2);
    
    // Game is over
    displayBoard(board, size);
    
    // Get final scores and display them
    compScore = userScore = 0;
    compScore = getScore(board, '@', size);
    userScore = getScore(board, 'O', size);
    fprintf(stdout,"Othello: the final score is:\n");
    fprintf(stdout,"Othello: Computer %d ------- User %d\n", compScore, userScore);
    if (userScore < compScore) {
        fprintf(stdout,"Othello: You lost, I won!!!\n");
    } else fprintf(stdout,"Othello: Congratulation!!!\n");
}
