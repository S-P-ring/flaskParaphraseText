import nltk

from itertools import permutations
from copy import deepcopy


def get_tree(tree_string: str) -> nltk.Tree:
    """Convert str tree to nltk.Tree object"""
    tree = nltk.tree.ParentedTree.fromstring(tree_string)
    return tree


def get_nps(tree: nltk.Tree) -> list[nltk.Tree]:
    """Get all NPs of the tree"""
    nps = [subtree for subtree in tree.subtrees() if subtree.label() == "NP"]
    return nps


def get_paraphrases_trees(np_groups: dict[tuple, list[nltk.Tree]], source_tree: nltk.Tree) -> list[nltk.Tree]:
    """Get all variants of paraphrases using this formula: n! / (n1! * n2! * ... * nk!)"""
    paraphrased_trees = []
    for parent_pos, child_nps in np_groups.items():
        child_positions = [np.treeposition()[-1] for np in child_nps]

        # generate all permutations of the child positions
        # use permutation instead of generator object, because input data isn`t great
        child_permutations = permutations(child_positions)

        # generate a new tree for each permutation
        for perm in child_permutations:
            # create a deep copy of the source tree to avoid modifying it
            new_tree = deepcopy(source_tree)

            # get the new positions of the child NPs based on the permutation
            new_child_positions = list(perm)
            new_child_nps = [deepcopy(new_tree[parent_pos][perm]) for perm in new_child_positions]

            # replace the original child NPs with the new ones
            for n, child_pos in enumerate(child_positions):
                new_tree[parent_pos][child_pos] = new_child_nps[n]

            # add the new paraphrased tree to the list
            paraphrased_trees.append(new_tree)

    return paraphrased_trees


def get_np_groups(nps: list[nltk.Tree]) -> dict[tuple, list[nltk.Tree]]:
    """
    Get all groups of NPs, that have ancestor NP
    np_groups is a dictionary of NP groups with the position of their parent NP
    """
    np_groups = {}
    for np in nps:
        # find the first common ancestor NP
        ancestor_np = np
        while ancestor_np.parent() is not None and ancestor_np.parent().label() == "NP":
            ancestor_np = ancestor_np.parent()

        # add the current NP to the group for this ancestor
        ancestor_pos = tuple(ancestor_np.treeposition())
        if ancestor_pos in np_groups:
            np_groups[ancestor_pos].append(np)
        else:
            np_groups[ancestor_pos] = [np]
    return np_groups


def output(data):
    output_data = {"paraphrases": []}

    for sentence in data:
        output_data["paraphrases"].append({
            "tree": str(sentence).replace('\n', '')
        })

    return output_data


def paraphrases(tree):
    data = get_paraphrases_trees(get_np_groups(get_nps(get_tree(tree))), get_tree(tree))
    return output(data)
