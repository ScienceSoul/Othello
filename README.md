# Othello

### Description:

This program plays the Othello game against a human opponent. The program provides also a playing mode in which an agent using a neural network can learn the game by playing against another agent.

* Language: C
* Compiler: Clang/LLVM
* Platform: masOS/Linux (posix threads)
* Required library: BLAS/LAPACK

Compilation:

```
cd src
make depend
make
```

Run with:

```
./Othello
```

The command above runs the game with the default settings: 8 x 8 board, human against computer (mode -play1) and a simple position evaluation function for the agent. One can give another board size and keep the same evaluation function with the command

```
./Othello 6
```

The agent can also use a minimax method to compute its next move. It is determined so that the opponent's best possible move score is a minimum, essentially exploring the nodes of the game tree up to depth one for each of the agent possible move. The command is as follows

```
./Othello 8 -minimax
```

When a method is explicitly given, it must follow the argument for the board size. Setting it alone is not currently supported. Also the simple position evaluation function can be chosen explicitly with the argument: -evaluation-function.

The program has also a neural network playing agent which can be trained with the TD-Learning algorithm. In order to test this agent and allow it to learn the game, it can play a user defined number of training games against another agent. The opponent agent can use either the evaluation function or the minimax method to play. The program can be run in this mode with the command

```
./Othello 8 -evaluation-function -play0
```

or

```
./Othello 8 -minimax -play0
```

Parameters used to control the learning algorithm are defined in ../params/parameters.dat. During the training session, weights and biases are saved and a human player can then play against the neural network playing agent with the command

```
./Othello 8 -neural-network
```

In this playing mode, the neural network playing agent will load the weights and biases computed from the training.

If another training session is performed, the program will load the existing data and continue the training from there. Once can start from scratch by simply deleting the files in the training directory. Currently it is not possible to have the neural network playing agent play the game against another agent without training at the same time.

Notes:

1. Posix threads are needed because the minimax method uses them so that one tread explores the game tree associated with one possible move of the agent, thereby all possible moves are processed in parallel.

2. Concerning the TD-Learning agent:
    1. Learning from self-play (both agents share the same neural network) is not currently supported.
    2. Learning from opponent’s moves is not currently supported, the agent only learns from its own moves.
    3. Actions are chosen using a \epsilon-greedy exploration, however the value of \epsilon is currently kept constant
during training

3. An implementation of BLAS/LAPACK is required to compile the program.

