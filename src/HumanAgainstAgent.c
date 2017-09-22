//
//  HumanAgainstAgent.c
//  Othello
//
//  Created by Seddik hakime on 31/05/2017.
//

#include "HumanAgainstAgent.h"
#include "NeuralAgent.h"

void humanAgainstAgent(char * _Nonnull * _Nonnull board, size_t size, int * _Nonnull * _Nonnull moves, char * _Nonnull  method) {
    
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
            fatal(PROGRAM_NAME, "failure reading input parameters.");
        }
        if (ntLayers[0] != (size*size)) {
            fatal(PROGRAM_NAME, "error in the network parameters. The number of input nodes should be equal to the board dimension row x col.");
        }
        if (ntLayers[numberOfLayers-1] != 1) {
            fatal(PROGRAM_NAME, "error in the network parameters. The number of output nodes should be one.");
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
            fprintf(stdout, "%s: number of moves so far: %d\n", PROGRAM_NAME, noOfMoves);
            
            if(player++ % 2) { // It is the player's turn
                if(validMoves(board, moves, 'O', size)) {
                    // Read player moves until a valid move is entered
                    int count = 0;
                    for(;;) {
                        fflush(stdin);                                                   // Flush the keyboard buffer
                        fprintf(stdout,"%s: please enter your move (row column): ", PROGRAM_NAME);
                        fscanf(stdin, "%d%c", &r, &y);                                   // Read input
                        c = tolower(y) - 'a';                                            // Convert to column index
                        r--;                                                             // Convert to row index
                        if(r>=0 && c>=0 && r<size && c<size && moves[r][c]) {
                            makeMove(board, r, c, 'O', size);
                            noOfMoves++;
                            break;
                        } else {
                            fprintf(stdout,"%s: not a valid move, try again.\n", PROGRAM_NAME);
                        }
                        // FIXME: Temporary work around when the loop goes infinite
                        // if the input is in the wrong format, e.g., c5
                        count++;
                        if (count > 10) {
                            fatal(PROGRAM_NAME, "something went wrong with the input.");
                        }
                    }
                } else {
                    // No valid moves
                    if(++invalidMoves<2) {
                        fflush(stdin);
                        fprintf(stdout,"\%s: you have to pass, press return:", PROGRAM_NAME);
                        fscanf(stdin, "%c", &retr);
                    } else {
                        fprintf(stdout,"\%s: neither of us can go, so the game is over.\n", PROGRAM_NAME);
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
                        neuralAgent(neural, board, size, moves, postState, ntLayers, numberOfLayers, '@', eta, lambda, gamma, epsilon, &newGame, false, NULL);
                    }
                    noOfMoves++;
                } else {
                    if(++invalidMoves<2) {
                        fprintf(stdout,"\%s: I have to pass, your go.\n", PROGRAM_NAME); // No valid move
                    }
                    else {
                        fprintf(stdout,"\%s: neither of us can go, so the game is over.\n", PROGRAM_NAME);
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
        fprintf(stdout,"%s: The final score is:\n", PROGRAM_NAME);
        fprintf(stdout,"%s: Agent %d ------- User %d\n", PROGRAM_NAME, agentScore, userScore);
        if (userScore < agentScore) {
            fprintf(stdout,"%s: You lost, I won.\n", PROGRAM_NAME);
        } else if (userScore > agentScore) {
            fprintf(stdout,"%s: Congratulation.\n", PROGRAM_NAME);
        } else fprintf(stdout,"%s: Draw.\n", PROGRAM_NAME);
        fprintf(stdout, "----------------------------------------------------------------------\n");
        
        fflush(stdin);
        fprintf(stdout, "Do you want to play again (y/n): ");
        fscanf(stdin, " %c", &again); // Get y or n
    
    } while(tolower(again) == 'y'); // Go again after inserting 'y'
    
    fprintf(stdout, "%s: end of game.\n", PROGRAM_NAME);
    
    if (strcmp(method, "-neural-network") == 0) {
        neural->destroy((void *)neural, NULL, false);
        free(neural);
        free_cmatrix(postState, 0, size-1, 0, size-1);
    }
}
