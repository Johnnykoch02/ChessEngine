from typing import List, Dict, Callable, Tuple, Optional
from CoderSchoolAI.Util.data_utils import dict_to_tensor
from CoderSchoolAI.Training.Algorithms import DictReplayBuffer
import torch as th
import numpy as np
import math
from src.Managers.RL.GrandMasterNetwork import GrandMasterValueAproximator

class MCTSNode:
    """
    We will be applying MCTS to our Chess Board Class.
    """
    def __init__(self, sim: 'MCTS', state, c_color, parent=None):
        self.parent = parent
        self.n_visit = 1
        self.sim = sim
        self.state = state # Board associated with this node
        self.c_color = c_color
        self.val = state.get_state(self.c_color)['score']
        self.children = {}
        
    def select_child(self,): # not a leaf
        children = list(self.children.values())
        return children[np.argmax([child.UCT(self.sim.C) for child in children])]
        
            
    def is_leaf(self,):
        return len(self.children == 0)
    
    def propagate(self, val):
        self.val += val
        if self.parent: self.parent.propagate(val)
    
    def UCT(self, C):
        return (self.val / self.n_visit) + C*math.sqrt(math.log(self.parent.n_visit) / self.n_visit)
    
    def simulate(self):
        '''
        a) Simulate to get the Value
        b) Create all possible Children
        c) return 
        '''
        v = 0
        for n_p in range(self.sim.n_playout):
            print('\tN Playout:', n_p)
            color = self.c_color
            c_board = self.state.create_virtual_board()
            for n_d in range(self.sim.depth):
                print('\t\tN Depth:', n_d)
                moves = set()
                for piece in c_board.g_pieces(color):
                    for place in self.sim.MoveGenerator.GenerateLegalMoves(piece, c_board)[0]:
                        moves.add((piece, place))  
                moves = list(moves)
                m = moves[np.random.randint(len(moves))]
                c_board = c_board.create_virtual_board()
                c_board.play_move(m[0], m[1])
                color = color >> color # Update the Color
                # c_board.
                if any(c_board.get_winner(color)): break  
            v += c_board.get_state(self.sim.team_color,)['score'][0]
        v /= self.sim.n_playout # Avg Val per playout
        self.propagate(v) # Adds to all Parents in Tree
        n_color = self.c_color >> self.c_color
        for piece in c_board.pieces(color): # Generate Children nodes
            for move in self.sim.MoveGenerator.GenerateLegalMoves(piece, c_board)[0]:
                n_board = self.c_board.create_virtual_board()
                n_board.play_move(piece, move)
                self.children[(piece, move)] = MCTSNode(self.sim, n_board, n_color, self,)
def GetSimulatedNodes(root: MCTSNode) -> List[MCTSNode]:
    def rec_traversal(c_node, memo):
        for child in c_node.children.values():
            if not child.is_leaf(): # the only useful Value Nodes come from Simulation
                rec_traversal(child, memo)
                memo.append(child)
        memo.append(c_node)
    memo = []
    rec_traversal(root, memo)
    return memo
    

class MCTS:
    """
    Class maintaining the search
    TODO: Implement Neural Network in Value Approximation for further fast implementation
    """
    def __init__(self, lr= 0.005, device='cuda:0'): 
        self.net = GrandMasterValueAproximator(None)
        self.device = th.device(device=device)
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=lr)
    
    def Simulate(self, init_state, team_color, env, MoveGenerator, n_playout=500, sim_depth=10, n_simulations=500, C=1.3, batch_size= 32) -> Tuple[object, ]:
        """
        This algorithm randomly simulates states throughout the tree to find the average value of each state. By sampling randomly, we assume that our estimated value converges to the actual value of a state, as random play has equal chance of picking optimal as well as picking poorly.
        init_state: Board, the initial state of the game
        team_color: Piece.Color, the team color of the game
        env: GrandMasterEnv, the environment being used.
        n_playout: int, the number of times we choose a branch to explore
        n_simulations: int, the number of simulations to run
        batch_size: int, number of elements per batch for the Neural Network
        """
        self.init_state = init_state
        self.team_color = team_color
        self.env = env
        self.MoveGenerator = MoveGenerator
        self.n_playout = n_playout
        self.depth = sim_depth
        
        self.C = C
        root = MCTSNode(self, init_state, team_color, team_color, )
        root.simulate()
        for _ in range(n_simulations):
            print('Sim:', n_simulations)
            c_node = root
            while not c_node.is_leaf():#TODO: self.n_visit +=1
                c_node.n_visit+=1
                c_node = c_node.select_child()
            c_node.simulate() # Simulates, creates new pathways, propagates value
        mse_loss = th.nn.MSELoss()
        sim_nodes = GetSimulatedNodes(root)
        replay = DictReplayBuffer(batch_size=batch_size)
        for s_node in sim_nodes:
            if len(replay.states) == replay.batch_size:
                states, _, _, vals, _, _ = replay.generate_batches() # S, A, P, V, R, D
                x = dict_to_tensor(states, self.device)
                self.net.train()
                self.optimizer.zero_grad()
                y = self.net(x)
                loss = mse_loss(y, vals)
                loss.backward()
                self.optimizer.step()
                replay.clear_memory()
                
            replay.store_memory(s_node.state.get_state(self.team_color), {'n': None}, None, s_node.val, 0, False) # Storing State and Value for now
         
        # Return Roots highest valued transition
        s_by = lambda tup: tup[1].val
        best = max([tup for tup in root.children.items()], s_by)
        return best[0]
            
                
            
            
            
        