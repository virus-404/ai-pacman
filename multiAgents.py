# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from node import Node
from util import manhattanDistance, Stack
from game import Actions, Directions, GameStateData

import random
import util

from game import Agent
from pacman import GameState

 
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total_score = 0
        old_food = currentGameState.getFood()
        for x in range(old_food.width):
            for y in range(old_food.height):
                if old_food[x][y]:
                    d = manhattanDistance((x, y), newPos)
                    if d == 0:
                        total_score += 100
                    else:
                        total_score += 1 / (d * d)

        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), newPos)
            if d <= 1:
                if ghost.scaredTimer != 0:
                    total_score += 2000
                else:
                    total_score -= 200

        return total_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def terminalTest(self, gameState, depth):
        return depth == 0 or gameState.isWin() or gameState.isLose()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        actions = []
        for a in gameState.getLegalActions(0):
            u = self.min_value(
                gameState=gameState.getNextState(0, a),
                agent=1,
                depth=self.depth,
            )
            if u == v:
                actions.append(a)
            elif u >= v:
                v = u
                actions = [a]

        #return random.choice(actions)
        return actions[0]

    def min_value(self, gameState: GameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, a)
            if agent == gameState.getNumAgents() - 1:
                v = min(
                    v, self.max_value(succ, agent=0, depth=depth - 1)
                )
            else:
                v = min(
                    v, self.min_value(succ, agent=agent + 1, depth=depth)
                )
        return v

    def max_value(self, gameState: GameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            v = max(
                v, self.min_value(gameState=gameState.getNextState(agent, a), 
                    agent=1, depth=depth)
            )
        return v
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***" 

        #return random.choice(actions)
        return self.max_value(
                        gameState=gameState,
                        agent=0,
                        depth=self.depth,
                        root=True)

    def max_value(self, gameState: GameState, agent, depth, alpha=float("-inf"), beta=float("+inf"), root=False):
        acc = []
        if self.terminalTest(gameState, depth):
            return acc if root else self.evaluationFunction(gameState)

        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            v = max(v, self.min_value(gameState=gameState.getNextState(agent, a),
                                      agent=1, depth=depth, alpha=alpha, beta=beta))
            if v > beta:
                return  a if root else v
            if alpha < v:
                alpha = v
                acc = a
        return acc if root else v
        
    def min_value(self, gameState: GameState, agent, depth, alpha, beta):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, a)
            if agent == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(succ, agent=0, depth=depth - 1, alpha=alpha, beta=beta))
            else:
                v = min(v, self.min_value(succ, agent=agent + 1, depth=depth, alpha=alpha, beta=beta)) 
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(
                    gameState=gameState,
                    agent=0,
                    depth=self.depth,
                    root=True)

    def max_value(self, gameState: GameState, agent, depth, root=False):
        act = []
        if self.terminalTest(gameState, depth):
            return act if root else self.evaluationFunction(gameState)
        
        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            succ =  gameState.getNextState(agent, a)
            u = self.chance(gameState = succ,
                            agent=1,
                            depth=depth)
            if v < u:
                v = u 
                act = a
        return act if root else v
        
    def chance(self, gameState: GameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        v=0
        prob_curr = 1 / len(gameState.getLegalActions(agent))

        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, a)
            if agent == gameState.getNumAgents() - 1:
                v += prob_curr * self.max_value(succ, agent=0, depth=depth - 1)
            else:
                v += prob_curr * self.chance(succ, agent=agent + 1, depth=depth)
        return v 

class MinimaxAgentIter(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3) iterative
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #return random.choice(actions)
        return self.iter(gameState=gameState,
                        agent=0,
                        depth=self.depth)

    def iter(self, gameState: GameState, agent, depth):
        stack = Stack()
        n = Node(parent=None, agent=agent, state=gameState, depth=depth)
        stack.push(n)
        
        while True:
            x = stack.top()
            if x.isRoot() and x.value is not None:
                return x.action
            if x.value is not None:
                if x.parent.isPacman() and  x.value > x.parent.value:
                    x.parent.value = x.value
                    x.parent.action = x.action
                elif not x.parent.isPacman() and x.value < x.parent.value:
                    x.parent.value = x.value
                stack.pop()    
            else:
                if self.terminalTest(x.state, x.depth):
                    x.value = self.evaluationFunction(x.state)
                else:
                    x.value = float("-inf") if x.isPacman() else float("+inf")
                    for a in x.state.getLegalActions(x.agent):
                        if x.isPacman():
                            x_child = Node(parent = x, agent = 1,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth)
                        else:
                            if x.agent == x.state.getNumAgents() - 1:
                                x_child = Node(parent = x, agent = 0,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth - 1)
                                
                            else: 
                                x_child = Node(parent = x, agent = x.agent + 1,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth)
                        x_child.action = a
                        stack.push(x_child)

class AlphaBetaAgentIter(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3) iterative
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #return random.choice(actions)
        return self.iter(gameState=gameState,
                        agent=0,
                        depth=self.depth)

    def iter(self, gameState: GameState, agent, depth):
        stack = Stack()
        n = Node(parent=None, agent=agent, state=gameState, depth=depth)
        n.alpha = float("-inf")
        n.beta = float("+inf")
        stack.push(n)
        
        while True:
            x = stack.top()
            if x.isRoot() and x.value is not None:
                return x.action
            if x.value is not None:
                if x.parent.isPacman():
                    if  x.value > x.parent.value: #Update my father
                        x.parent.value = x.value
                        x.parent.action = x.action
                    if x.value < x.alpha:         #Kill my brothers *evil laughs* (kinda sat... send me an @ if you like it)
                        stack.pop()
                        while x.parent == stack.top().parent:
                            stack.pop()
                        stack.push(x)
                    x.parent.beta = min(x.parent.beta, x.value)
                   
                else:
                    if x.value < x.parent.value:
                        x.parent.value = x.value
                    if x.value > x.parent.beta: 
                        stack.pop()
                        while x.parent == stack.top().parent:
                            stack.pop()
                        stack.push(x)
                    x.parent.alpha = max(x.parent.alpha, x.value)
                stack.pop()

            else:
                if self.terminalTest(x.state, x.depth):
                    x.value = self.evaluationFunction(x.state)
                else:
                    x.value = float("-inf") if x.isPacman() else float("+inf")
                    for a in x.state.getLegalActions(x.agent):
                        if x.isPacman():
                            x_child = Node(parent = x, agent = 1,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth)
                        else:
                            if x.agent == x.state.getNumAgents() - 1:
                                x_child = Node(parent = x, agent = 0,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth - 1)
                                
                            else: 
                                x_child = Node(parent = x, agent = x.agent + 1,\
                                            state = x.state.getNextState(x.agent, a),\
                                            depth=x.depth)
                        x_child.action = a
                        x_child.alpha = x.alpha
                        x_child.beta = x.beta
                        stack.push(x_child)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    total_score = -currentGameState.getNumFood()*2

    food = currentGameState.getFood()
    pacman = currentGameState.getPacmanPosition()
    
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                d = manhattanDistance((x, y), pacman)
                if d == 0:
                    total_score += 100
                else:
                    total_score += 1 / d

    for idx in range(1, currentGameState.getNumAgents()):
        d = manhattanDistance(currentGameState.getGhostPosition(idx), pacman)
        capsule_d = [manhattanDistance(pacman, cap)for cap in currentGameState.getCapsules()]
        if len(capsule_d) > 0:
            if d > capsule_d[0] and d < 3:
                total_score += 2000
        if d < 2 and currentGameState.getGhostState(idx).scaredTimer != 0:
            total_score += 500
        if d >= 1:
            total_score += 50
    return total_score 

# Abbreviation
better = betterEvaluationFunction
