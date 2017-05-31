//
//  main.c
//  Othello
//
//  Created by Seddik hakime on 24/05/2017.
//

#include "HumanAgainstAgent.h"

int main(int argc, const char * argv[]) {

    size_t size = 0;          // Board size
    char retr = 0;            // Return key

    // Agent moving method: -minimax or -evaluation-function
    char *method = (char *)malloc(64*sizeof(char));
    memset(method, 0, 64*sizeof(char));
    
    // TODO: better input management. Currently can't only give
    // the method without also giving the board size
    if (argc < 2 || argc == 2) memcpy(method, "-evaluation-function", strlen("-evaluation-function")*sizeof(char));
    if (argc < 2) {
        size = 8;
    } else if (argc == 2) {
        size = atoi(argv[1]);
    } else if (argc == 3) {
        size = atoi(argv[1]);
        memcpy(method, argv[2], strlen(argv[2])*sizeof(char));
        if (strcmp(method, "-minimax") != 0 && strcmp(method, "-evaluation-function") != 0) {
            fatal("Othello", "unknown method to compute the agant moves.");
        }
    } else if (argc > 3) {
        fatal("Othello", " number of input argument must at most 2.");
    }
    
    // Board size - must be even
    if (size % 2 != 0) {
        fatal("Othello", "board size should be a multiple of 2.");
    }
    
    // The board
    char **board = charmatrix(0, size-1, 0, size-1);
    memset(*board, 0, (size*size)*sizeof(char));
    
    // Valid moves
    int **moves = intmatrix(0, size-1, 0, size-1);
    memset(*moves, 0, (size*size)*sizeof(int));

    fprintf(stdout,"\nOthello: Othello Game. Starting now...\n\n");
    fprintf(stdout,"Othello: You will play first.\n");
    fprintf(stdout,"Othello: You will be white - (O). I will be black - (@).\n");
    fprintf(stdout,"Othello: Select a square for your move by typing a digit for the row\n "
                   "        and a letter for the column with no spaces between.\n");
    fprintf(stdout,"Othello: Agent playing method: %s.\n", method);
    fprintf(stdout,"\nOthello: Press Enter to start.\n");
    fscanf(stdin, "%c", &retr);
    
    humanAgainstAgent(board, size, moves, method);
    
    free_cmatrix(board, 0, size-1, 0, size-1);
    free_imatrix(moves, 0, size-1, 0, size-1);
    free(method);

    return 0;
}
