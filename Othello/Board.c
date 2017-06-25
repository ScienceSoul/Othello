//
//  Board.c
//  Othello
//
//  Created by Seddik hakime on 25/05/2017.
//

#include "Board.h"

/***********************************************
 
    Function to display the board in it's
    current state with row numbers and column
    letters to identify squares.
 
************************************************/
void displayBoard(char * _Nonnull * _Nonnull board, size_t size) {
    
    char colLabel = 'a';   // Column label
    
    fprintf(stdout,"\n ");                        // Start top line
    for(int col=0; col<size; col++)
        fprintf(stdout,"   %c", colLabel+col);    // Display the top
    
    fprintf(stdout,"\n");                         // End the top line
    
    // Display the intermediate rows
    for(int row=0; row<size; row++) {
        fprintf(stdout,"  +");
        for(int col=0; col<size; col++)
            fprintf(stdout,"---+");
        
        fprintf(stdout,"\n%2d|", row+1);
        
        for(int col=0; col<size; col++)
            fprintf(stdout," %c |", board[row][col]);  // Display counters in row
        
        fprintf(stdout,"\n");
    }
    
    fprintf(stdout,"  +");                     // Start the bottom line
    for(int col = 0 ; col<size ;col++)
        fprintf(stdout,"---+");                // Display the bottom line
    
    fprintf(stdout,"\n");                      // End the bottom  line
}
