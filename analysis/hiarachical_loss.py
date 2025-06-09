import torch
import sys
import time
import copy
import numpy as np
import gc
import networkx as nx


class HierarchicalLoss():
    def __init__(self, hierarchy, device):
        self.hierarchy = hierarchy
        self.hierarchy_graph = self.build_hierarchy_graph(hierarchy)
        self.T32, self.T21, self.leaf_to_idx, self.mid_to_idx, self.top_to_idx = self.generate_transition_matrices(self.hierarchy_graph)   
        self.T32 = torch.from_numpy(self.T32).to(device)
        self.T21 = torch.from_numpy(self.T21).to(device)
        self.alpha = 0.5  # Default value, should be set appropriately

    def add_edges_from_hierarchy(self, graph, parent):
        if parent in self.hierarchy:
            for child in self.hierarchy[parent]:
                graph.add_edge(parent, child)
                self.add_edges_from_hierarchy(graph, child)    
                
    def build_hierarchy_graph(self, hierarchy):
        graph = nx.DiGraph()
        for parent in hierarchy:
            self.add_edges_from_hierarchy(graph, parent)
        return graph            
    
    def generate_transition_matrices(self, G):
        """
        Generate transition matrices from a hierarchical graph structure.
        
        Args:
            G: A NetworkX DiGraph representing a hierarchical tree
            
        Returns:
            tuple: (T32, T21, leaf_to_idx, mid_to_idx, top_to_idx)
                - T32: Transition matrix from leaf to mid-level nodes
                - T21: Transition matrix from mid to top-level nodes
                - leaf_to_idx: Dictionary mapping leaf nodes to indices
                - mid_to_idx: Dictionary mapping mid-level nodes to indices
                - top_to_idx: Dictionary mapping top-level nodes to indices
        """
        # Find the root node
        roots = [n for n, d in G.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root; found {roots}")
        root = roots[0]
        
        # Compute depth of each node using BFS
        depth = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in G.successors(node):
                if child not in depth:
                    depth[child] = depth[node] + 1
                    queue.append(child)
        
        # Organize nodes by depth
        nodes_by_depth = {}
        for node, d in depth.items():
            nodes_by_depth.setdefault(d, []).append(node)
        
        # Extract nodes at each level
        top_nodes = sorted(nodes_by_depth.get(1, []))
        mid_nodes = sorted(nodes_by_depth.get(2, []))
        leaf_nodes = sorted(nodes_by_depth.get(3, []))
        
        # Create index mappings
        leaf_to_idx = {leaf: i for i, leaf in enumerate(leaf_nodes)}
        mid_to_idx = {mid: i for i, mid in enumerate(mid_nodes)}
        top_to_idx = {top: i for i, top in enumerate(top_nodes)}
        
        # Build T32 matrix (leaf → mid)
        num_leaves = len(leaf_nodes)
        num_mid = len(mid_nodes)
        T32 = np.zeros((num_leaves, num_mid), dtype=np.float32)
        
        for leaf in leaf_nodes:
            parents = list(G.predecessors(leaf))
            if len(parents) != 1 or parents[0] not in mid_to_idx:
                raise ValueError(f"Leaf '{leaf}' should have exactly one mid-node parent, but found {parents}.")
            parent_mid = parents[0]
            i = leaf_to_idx[leaf]
            j = mid_to_idx[parent_mid]
            T32[i, j] = 1.0
        
        # Build T21 matrix (mid → top)
        num_top = len(top_nodes)
        T21 = np.zeros((num_mid, num_top), dtype=np.float32)
        
        for mid in mid_nodes:
            parents = list(G.predecessors(mid))
            if len(parents) != 1 or parents[0] not in top_to_idx:
                raise ValueError(f"Mid-node '{mid}' should have exactly one top-node parent, but found {parents}.")
            parent_top = parents[0]
            j = mid_to_idx[mid]
            k = top_to_idx[parent_top]
            T21[j, k] = 1.0
        
        return T32, T21, leaf_to_idx, mid_to_idx, top_to_idx

    def get_loss(self, prediction, one_hot_label):
        """
        Calculate the hierarchical loss based on the prediction and one-hot encoded label,
        weighting each conditional negative log‐probability by exp(−α·ℓ) as in Eq.(4) of the paper.
        Assumes that `self.alpha` has been set (e.g. in __init__) to the chosen α>0.
        """
        # 1) Leaf‐level probabilities (softmax over logits)
        leaf_prob = torch.softmax(prediction, dim=-1)  # [B, #leaves]

        # 2) One‐hot targets for mid‐ and top‐levels
        mid_one_hot = torch.matmul(one_hot_label, self.T32)  # [B, #mid]
        top_one_hot = torch.matmul(mid_one_hot, self.T21)     # [B, #top]

        # 3) Predicted probabilities for mid‐ and top‐levels
        mid_prob = torch.matmul(leaf_prob, self.T32)  # [B, #mid]
        top_prob = torch.matmul(mid_prob,   self.T21)  # [B, #top]

        # 4) Negative log‐likelihood (cross‐entropy) at each level
        #    Add a small epsilon inside log to avoid numerical issues if prob=0.
        eps = 1e-12
        leaf_loss = -torch.sum(one_hot_label  * torch.log(leaf_prob + eps), dim=-1)  # L₃
        mid_loss  = -torch.sum(mid_one_hot    * torch.log(mid_prob   + eps), dim=-1)  # L₂
        top_loss  = -torch.sum(top_one_hot    * torch.log(top_prob   + eps), dim=-1)  # L₁

        # 5) Compute exponential weights λₗ = exp(−α·ℓ) for ℓ = 1,2,3
        a = self.alpha
        λ1 = torch.exp(torch.tensor(-a * 1.0))  # weight for −log p(C^(1)|C^(0)) = L₁
        λ2 = torch.exp(torch.tensor(-a * 2.0))  # weight for −log p(C^(2)|C^(1)) = (L₂ − L₁)
        λ3 = torch.exp(torch.tensor(-a * 3.0))  # weight for −log p(C^(3)|C^(2)) = (L₃ − L₂)

        # 6) Build the "conditional" losses
        cond_32 = leaf_loss - mid_loss  # = −log p(C^(3)|C^(2))
        cond_21 = mid_loss  - top_loss  # = −log p(C^(2)|C^(1))
        top_only = top_loss             # = −log p(C^(1)|C^(0))

        total_loss = λ3 * cond_32 \
                   + λ2 * cond_21 \
                   + λ1 * top_only

        return total_loss.mean()



