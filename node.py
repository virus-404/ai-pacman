class Node: 

    def __init__(self, parent, agent, state, depth=0):
        self.parent = parent
        self.agent = agent
        self.state = state
        self.depth = depth
        self.action = []
        self.value = None
        self.alpha = None
        self.beta = None

    def isMiniMax(self):
        return self.alpha == None
    
    def isPacman(self):
        return self.agent == 0

    def isRoot(self):
        return self.parent == None
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.parent == other.parent \
                    and self.agent == other.agent\
                    and self.state == other.state\
                    and self.depth == other.depth\
                    and self.value == other.value
        return False

    