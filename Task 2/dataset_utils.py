
import re
import random

class BinaryTree:
    def __init__(self, tree_string):
        """
        Initializes the tree from a string in the format "parent>child,parent>child,..."
        """
        self.nodes = {}
        # Splits the string into parent>child pairs
        pairs = [p.strip() for p in tree_string.split(',') if '>' in p]
        
        for pair in pairs:
            parent, child = pair.split('>')
            parent, child = parent.strip(), child.strip()
            
            if parent not in self.nodes:
                self.nodes[parent] = []
            # Adds the child to the parent node's list of children
            self.nodes[parent].append(child)

    def get_children(self, node):
        """
        Given a node, returns its children (up to two).
        Returns an empty list if the node has no children or does not exist.
        """
        return self.nodes.get(str(node), [])

    def get_max_depth(self, node):
        """
        Given a node, returns the maximum depth of the subtree
        rooted at that node (if the node is a leaf, the depth is 0).
        """
        node_str = str(node)
        children = self.get_children(node_str)
        
        if not children:
            return 0
        
        # Recursively computes the maximum depth of the child subtrees
        return 1 + max(self.get_max_depth(child) for child in children)

    def dfs(self,node, target, depth):
        if node == target:
            return depth
        children = self.get_children(node)
        for child in children:
            result = self.dfs(child, target, depth + 1)
            if result != -1:
                return result
        return -1

    def get_depth_to_leaf(self, root, leaf):
        return self.dfs(str(root), str(leaf), 0)

    def check_connectivity(self, root, leaf):
        """
        Checks whether a path exists from the root to the leaf in the tree.
        Returns True if the path exists, and False otherwise.
        """
        return self.dfs(str(root), str(leaf), 0) != -1


def remap_line(line,vocal_size,n_states):
    pattern = re.compile(r'(?<![0-9])(\d{1,2})(?![0-9])')
    
    targets = random.sample(range(vocal_size), n_states)
    mapping = {i: targets[i] for i in range(n_states)}
    
    def repl(m):
        val = int(m.group(0))
        if 0 <= val < n_states:
            return str(mapping[val])
        return m.group(0)
    
    return pattern.sub(repl, line)

def compute_dataset_depths(file_path):
    """
    For each line in the file:
    1) uses the part to the left of '|' to build a BinaryTree
    2) uses the part to the right of '|' to identify the leaf and root (separated by ':')
        
    Returns a depth value for each line stored in a list.
    """
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            tree_part, rest = line.split('|')
            
            node_part = rest.split('>')[0]
            
            # Extract the leaf and root (format: leaf:root)
            parts = node_part.split(':')
            if len(parts) != 2:
                continue
            
            leaf, root = parts[0], parts[1]
            
            # Build the tree
            tree = BinaryTree(tree_part)
            
            # Compute the depth from the root to the leaf
            depth = tree.get_depth_to_leaf(root, leaf)
            results.append(depth)
                
    return results

def remove_link_from_tree(tree, node_to_break,tree_part):
    #find the parent of the node to break
    parent_node = None
    for parent, children in tree.nodes.items():
        if node_to_break in children:
            parent_node = parent
            break
    
    # Remove the link between the node and its parent from the tree_part string
    tree_part = tree_part.replace(f",{parent_node}>{node_to_break},", ",")  #gestisce il caso in cui il nodo da rimuovere è in mezzo alla stringa
    
    if tree_part.startswith(f"{parent_node}>{node_to_break},"): #gestisce il caso in cui il nodo da rimuovere è all'inizio della stringa
        tree_part = tree_part[len(f"{parent_node}>{node_to_break},"):]

    if tree_part.endswith(f",{parent_node}>{node_to_break}"): #gestisce il caso in cui il nodo da rimuovere è alla fine della stringa
        tree_part = tree_part[:-len(f",{parent_node}>{node_to_break}")]

    return tree_part

def break_tree(data_string, verbose=False):
    tree_part, path_part = data_string.split('|')
    leaf,path_string = path_part.split(':')

    tree = BinaryTree(tree_part)

    splitted_path = path_string.split('>')

    break_path = random.choice([True, False])

    or_tree_part_impl = tree_part.count('>')

    new_data_string = None
    if break_path:
        # Randomly select a node from the path, excluding the first and last nodes
        if len(splitted_path) > 2:
            node_to_break = random.choice(splitted_path[1:-1])
            if verbose:
                print(f"Positive break: Node '{node_to_break}' will be removed from the tree.")
            tree_part =remove_link_from_tree(tree, node_to_break, tree_part)

            if tree_part.count('>') != or_tree_part_impl-1:
                print(f"Error: tree_part has {tree_part.count('>')} '>' characters instead of {or_tree_part_impl-1}")

            new_data_string = f"{tree_part}|{leaf}:{splitted_path[0]}N"
    else:
        # Create a list of all non-leaf nodes in the tree
        non_leaf_nodes = [node for node in tree.nodes.keys() if tree.get_children(node)]
        # Exclude non-leaf nodes that are already in the path
        non_leaf_nodes = [node for node in non_leaf_nodes if node not in splitted_path]
        if len(non_leaf_nodes)>0:
            # Randomly select one of the non-leaf nodes and print it
            node_to_break = random.choice(non_leaf_nodes)
            if verbose:
                print(f"Negative break: Node '{node_to_break}' will be removed from the tree.")
            tree_part =remove_link_from_tree(tree, node_to_break, tree_part)

            if tree_part.count('>') != or_tree_part_impl-1:
                print(f"Error: tree_part has {tree_part.count('>')} '>' characters instead of {or_tree_part_impl-1}")

            new_data_string = f"{tree_part}|{leaf}:{splitted_path[0]}Y"

    return new_data_string

def check_double_root_in_tree(tree_part,verbose=False):
    new_tree = BinaryTree(tree_part)
    nodes_without_parent = []
    for node in new_tree.nodes.keys():
        has_parent = False
        for children in new_tree.nodes.values():
            if node in children:
                has_parent = True
                break
        if not has_parent:
            nodes_without_parent.append(node)
    if verbose:
        print("Nodes without parent in the new tree:", nodes_without_parent)

    return len(nodes_without_parent) == 2