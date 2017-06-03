//
//  HumanAgainstAgent.c
//  Othello
//
//  Created by Seddik hakime on 31/05/2017.
//

#include "HumanAgainstAgent.h"
#include "NeuralAgent.h"

void humanAgainstAgent(char * __nonnull * __nonnull board, size_t size, int * __nonnull * __nonnull moves, char * __nonnull  method) {
    
    int noOfGames = 0;                // Number of games
    int noOfMoves = 0;                // Count of moves
    int invalidMoves = 0;             // Invalid move count
    int agentScore = 0;               // Computer score
    int userScore = 0;                // Player score
    char y = 0;                       // Column letter
    int c = 0;                        // Column number
    int r = 0;                        // Row number
    char retr = 0;                    // Return key
    char again = 0;                   // Replay choice
    int player = 0;                   // Player indicator
    bool newGame;
    
    NeuralNetwork *neural = NULL;
    size_t numberOfGame=0;
    int ntLayers[100];
    size_t numberOfLayers=0;
    char **postState = NULL;
    float eta=0, lambda=0, gamma=0, epsilon=0;
    if (strcmp(method, "-neural-network") == 0) {
        neural = allocateNeuralNetwork();
        
        memset(ntLayers, 0, sizeof(ntLayers));
        if (loadParameters(ntLayers, &numberOfLayers, &eta, &lambda, &gamma, &epsilon, &numberOfGame) != 0) {
            fatal("Othello", "failure reading input parameters.");
        }
        if (ntLayers[0] != (size*size)) {
            fatal("Othello", "error in the network parameters. The number of input nodes should be equal to the board dimension row x col.");
        }
        if (ntLayers[numberOfLayers-1] != 1) {
            fatal("Othello", "error in the network parameters. The number of output nodes should be one.");
        }
        
        neural->create((void *)neural, ntLayers, numberOfLayers, NULL, false);
        postState = charmatrix(0, size-1, 0, size-1);
    }
    
    // The main game loop
    do {
        
        fprintf(stdout, "\n***************************************\n");
        fprintf(stdout, "New game:\n");
        fprintf(stdout, "***************************************\n");
    
        // On even games the player starts;
        // On odd games the agent starts
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
        
        newGame = true;
        
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
                        fprintf(stdout,"\nOthello: you have to pass, press return:");
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
                    if (strcmp(method, "-minimax") == 0 || strcmp(method, "-evaluation-function") == 0) {
                        agent(board, moves, numberOfMoves, '@', size, method);
                    } else {
                        neuralAgent(neural, board, size, moves, postState, ntLayers, numberOfLayers, '@', eta, lambda, gamma, epsilon, &newGame);
                    }
                    noOfMoves++;
                } else {
                    if(++invalidMoves<2) {
                        fprintf(stdout,"\nOthello: I have to pass, your go.\n"); // No valid move
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
        agentScore = userScore = 0;
        agentScore = getScore(board, '@', size);
        userScore = getScore(board, 'O', size);
        fprintf(stdout, "----------------------------------------------------------------------\n");
        fprintf(stdout,"Othello: the final score is:\n");
        fprintf(stdout,"Othello: Agent %d ------- User %d\n", agentScore, userScore);
        if (userScore < agentScore) {
            fprintf(stdout,"Othello: You lost, I won!!!\n");
        } else fprintf(stdout,"Othello: Congratulation!!!\n");
        fprintf(stdout, "----------------------------------------------------------------------\n");
        
        fflush(stdin);
        printf("Do you want to play again (y/n): ");
        scanf("%c", &again); // Get y or n
    
    } while(tolower(again) == 'y'); // Go again after inserting 'y'
    
    if (strcmp(method, "-neural-network") == 0) {
        neural->destroy((void *)neural, NULL, false);
        free(neural);
        free_cmatrix(postState, 0, size-1, 0, size-1);
    }
}
