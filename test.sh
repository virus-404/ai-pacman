echo "MiniMax";
python3 pacman.py -p MinimaxAgent -l $1 -a depth=2  -q -n $2 | grep Average;
echo "AlphaBeta";
python3 pacman.py -p AlphaBetaAgent -l $1 -a depth=2  -q -n $2 | grep Average;
echo "Expectimax";
python3 pacman.py -p ExpectimaxAgent -l $1 -a depth=2  -q -n $2 | grep Average;

# -q: without graphical interface
# -n <N>: number of games

# MiniMax (10 executions with default evaluation function)
#python3 pacman.py -p MinimaxAgent -l smallClassic -a depth=2 -q -n 10

# MiniMax (10 executions with better evaluation function)
#python3 pacman.py -p MinimaxAgent -l smallClassic -a depth=2,evalFn=better -q -n 10