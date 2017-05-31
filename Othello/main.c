//
//  main.c
//  Othello
//
//  Created by Seddik hakime on 24/05/2017.
//

#include <stdio.h>
#include <ctype.h>

#include "Board.h"
#include "Move.h"
#include "Memory.h"

int main(int argc, const char * argv[]) {

    size_t SIZE = 0;                     // Board size
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
    
    // Agent moving method: -minimax or -evaluation-function
    char *method = (char *)malloc(64*sizeof(char));
    memset(method, 0, 64*sizeof(char));
    
    // TODO: better input management. Currently can't only give
    // the method without also giving the board size
    if (argc < 2 || argc == 2) memcpy(method, "-evaluation-function", strlen("-evaluation-function")*sizeof(char));
    if (argc < 2) {
        SIZE = 8;
    } else if (argc == 2) {
        SIZE = atoi(argv[1]);
    } else if (argc == 3) {
        SIZE = atoi(argv[1]);
        memcpy(method, argv[2], strlen(argv[2])*sizeof(char));
        if (strcmp(method, "-minimax") != 0 && strcmp(method, "-evaluation-function") != 0) {
            fatal("Othello", "unknown method to compute the agant moves.");
        }
    } else if (argc > 3) {
        fatal("Othello", " number of input argument must at most 2.");
    }
    
    // Board size - must be even
    if (SIZE % 2 != 0) {
        fatal("Othello", "board size should be a multiple of 2.");
    }
    
    // The board
    char **board = charmatrix(0, SIZE-1, 0, SIZE-1);
    memset(*board, 0, (SIZE*SIZE)*sizeof(char));
    
    // Valid moves
    int **moves = intmatrix(0, SIZE-1, 0, SIZE-1);
    memset(*moves, 0, (SIZE*SIZE)*sizeof(int));

    fprintf(stdout,"\nOthello: Othello Game. Starting now...\n\n");
    fprintf(stdout,"Othello: You will play first.\n");
    fprintf(stdout,"Othello: You will be white - (O). I will be black - (@).\n");
    fprintf(stdout,"Othello: Select a square for your move by typing a digit for the row\n "
                   "        and a letter for the column with no spaces between.\n");
    fprintf(stdout,"Othello: Agent playing method: %s.\n", method);
    fprintf(stdout,"\nOthello: Press Enter to start.\n");
    fscanf(stdin, "%c", &retr);

    // On even games the player starts;
    // On odd games the computer starts
    player = ++noOfGames % 2;
    
    // Starts with four for the move count
    noOfMoves = 4;
    
    // Blank all the board squares */
    for(int row = 0; row < SIZE; row++) {
        for(int col = 0; col < SIZE; col++) {
            board[row][col] = ' ';
        }
    }
    
    // Place the initial four counters in the center
    board[SIZE/2 - 1][SIZE/2 - 1] = board[SIZE/2][SIZE/2] = 'O';
    board[SIZE/2 - 1][SIZE/2]     = board[SIZE/2][SIZE/2 - 1] = '@';
    
    // The game play loop
    do {
        
        displayBoard(board, SIZE);
        fprintf(stdout, "Othello: number of moves so far: %d\n", noOfMoves);
        
        if(player++ % 2) { // It is the player's turn
            if(validMoves(board, moves, 'O', SIZE)) {
                // Read player moves until a valid move is entered
                int count = 0;
                for(;;) {
                    fflush(stdin);                                                   // Flush the keyboard buffer
                    fprintf(stdout,"Othello: please enter your move (row column): ");
                    fscanf(stdin, "%d%c", &r, &y);                                   // Read input
                    c = tolower(y) - 'a';                                            // Convert to column index
                    r--;                                                             // Convert to row index
                    if(r>=0 && c>=0 && r<SIZE && c<SIZE && moves[r][c]) {
                        makeMove(board, r, c, 'O', SIZE);
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
            int numberOfMoves = validMoves(board, moves, '@', SIZE);
            if(numberOfMoves) {
                // Reset invalid count
                invalidMoves = 0;
                agent(board, moves, numberOfMoves, '@', SIZE, method);
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
    } while(noOfMoves < SIZE*SIZE && invalidMoves<2);
    
    // Game is over
    displayBoard(board, SIZE);
    
    // Get final scores and display them
    compScore = userScore = 0;
    compScore = getScore(board, '@', SIZE);
    userScore = getScore(board, 'O', SIZE);
    fprintf(stdout,"Othello: the final score is:\n");
    fprintf(stdout,"Othello: Computer %d ------- User %d\n", compScore, userScore);
    if (userScore < compScore) {
        fprintf(stdout,"Othello: You lost, I won!!!\n");
    } else fprintf(stdout,"Othello: Congratulation!!!\n");
    
    free_cmatrix(board, 0, SIZE-1, 0, SIZE-1);
    free_imatrix(moves, 0, SIZE-1, 0, SIZE-1);
    free(method);

    return 0;
}
