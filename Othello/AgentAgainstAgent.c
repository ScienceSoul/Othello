//
//  AgentAgainstAgent.c
//  Othello
//
//  Created by Seddik hakime on 02/06/2017.
//

#include <dirent.h>
#include "AgentAgainstAgent.h"
#include "NeuralAgent.h"

void agentAgainstAgent(char * __nonnull * __nonnull board, size_t size, int * __nonnull * __nonnull moves, char * __nonnull  method) {
    
    int noOfGames = 0;                // Number of games
    int noOfMoves = 0;                // Count of moves
    int invalidMoves = 0;             // Invalid move count
    int agentScore = 0;               // Computer score
    int opponentScore = 0;            // Player score
    int player = 0;                   // Player indicator
    bool newGame;
    bool alreadyTrained = true;

    NeuralNetwork *neural = NULL;
    size_t numberOfGames=0;
    int ntLayers[100];
    size_t numberOfLayers=0;
    char **postState = NULL;
    float eta=0, lambda=0, gamma=0, epsilon=0;
    
    neural = allocateNeuralNetwork();
    
    memset(ntLayers, 0, sizeof(ntLayers));
    if (loadParameters(ntLayers, &numberOfLayers, &eta, &lambda, &gamma, &epsilon, &numberOfGames) != 0) {
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
    
    int numberOfNeuralAgentVictories = 0;
    for (int g=1; g<=numberOfGames; g++) {
        
        fprintf(stdout, "\n***************************************\n");
        fprintf(stdout, "Playing game %d:\n", g);
        fprintf(stdout, "***************************************\n");
        
        // On even games the opponent agent starts;
        // On odd games the neural agent starts
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
            
            if(player++ % 2) { // It is the agent turn
                int numberOfMoves = validMoves(board, moves, 'O', size);
                if(numberOfMoves) {
                    agent(board, moves, numberOfMoves, 'O', size, method);
                    noOfMoves++;
                } else {
                    if(++invalidMoves<2) {
                        fprintf(stdout,"\nOthello: opponent agent has to pass.\n"); // No valid move
                    }
                    else {
                        fprintf(stdout,"\nOthello: neither of agents can go, so the game is over.\n");
                        break;
                    }
                }
            } else { // It is the neural agent turn
                int numberOfMoves = validMoves(board, moves, '@', size);
                if(numberOfMoves) {
                    // Reset invalid count
                    invalidMoves = 0;
                    neuralAgent(neural, board, size, moves, postState, ntLayers, numberOfLayers, '@', eta, lambda, gamma, epsilon, &newGame, true, &alreadyTrained);
                    noOfMoves++;
                } else {
                    if(++invalidMoves<2) {
                        fprintf(stdout,"\nOthello: meural agent has to pass.\n"); // No valid move
                    }
                    else {
                        fprintf(stdout,"\nOthello: neither of agents can go, so the game is over.\n");
                        break;
                    }
                }
            }
        } while(noOfMoves < size*size && invalidMoves<2);
        
        // Game is over
        displayBoard(board, size);
        
        // Get final scores and display them
        agentScore = opponentScore = 0;
        agentScore = getScore(board, '@', size);
        opponentScore = getScore(board, 'O', size);
        if (agentScore > opponentScore) numberOfNeuralAgentVictories++;
        fprintf(stdout, "----------------------------------------------------------------------\n");
        fprintf(stdout,"Othello: the final score is:\n");
        fprintf(stdout,"Othello: Neural agent %d ------- Agent %d\n", agentScore, opponentScore);
        fprintf(stdout, "----------------------------------------------------------------------\n");
    }
    fprintf(stdout, "Othello: number of neural agent victories during training: %d/%zu\n", numberOfNeuralAgentVictories, numberOfGames);
    fprintf(stdout, "----------------------------------------------------------------------\n");
    
    // Store the weights and biases
    storeWeightsAndBiases((void *)neural, ntLayers, numberOfLayers);
    
    neural->destroy((void *)neural, NULL, false);
    free(neural);
    free_cmatrix(postState, 0, size-1, 0, size-1);
}
