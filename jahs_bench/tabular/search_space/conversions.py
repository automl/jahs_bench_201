
"""
There are three representations
'naslib': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper

This file currently has the following conversions:
naslib -> op_indices
op_indices -> naslib
naslib -> arch_str

Note: we could add more conversions, but this is all we need for now
"""

import ConfigSpace
from .constants import OP_NAMES, EDGE_LIST, nb201_to_ops, ops_to_nb201

def convert_naslib_to_op_indices(naslib_object):

    cell = naslib_object._get_child_graphs(single_instances=True)[0]
    ops = []
    for i, j in EDGE_LIST:
        ops.append(cell.edges[i, j]['op'].get_op_name)

    return [OP_NAMES.index(name) for name in ops]


def convert_op_indices_to_naslib(op_indices, naslib_object):
    """
    Converts op indices to a naslib object
    input: op_indices (list of six ints)
    naslib_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a naslib object that has already been 
    discretized (i.e., all edges have a single op).

    output: none, but the naslib object now has all edges set
    as in genotype.
    
    warning: this method will modify the edges in naslib_object.
    """
    
    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = OP_NAMES[index]
    
    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                if op.get_op_name == edge_op_dict[(edge.head, edge.tail)]:
                    index = i
                    break
            edge.data.set('op_index', index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        edge.data.set('op', primitives[edge.data.op_index])
        edge.data.set('primitives', primitives)     # store for later use

    naslib_object.update_edges(
        add_op_index,
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=False
    )
    
    naslib_object.update_edges(
        update_ops, 
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=True
    )

def convert_naslib_to_str(naslib_object):
    """
    Converts naslib object to string representation.
    """

    cell = naslib_object.edges[2, 3].op
    edge_op_dict = {
        (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
    }
    op_edge_list = [
        '{}~{}'.format(edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)


def convert_str_to_indices(arch_str: str) -> list:
    """ Converts a given string denoting the cell architecture in the format used by the original NASBench-201 data
    into a list of indices that can be used by NASLib NASB201HPOSearchSpace objects. """

    nodestrs = arch_str.split('+')
    edge_ops = {k: None for k in EDGE_LIST}
    out_node = 2
    for i, node_str in enumerate(nodestrs):
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = (xi.split('~') for xi in inputs)
        try:
            input_infos = tuple((nb201_to_ops[nb201_op], int(IDX) + 1) for (nb201_op, IDX) in inputs)
        except KeyError as e:
            raise RuntimeError("Failed to translate the given NB201 architecture string to a NASLib architecture "
                               "description.") from e
        for op, in_node in input_infos:
            edge_name = in_node, out_node
            if edge_name not in edge_ops:
                raise ValueError(f"The given architecture string requires an edge at {edge_name}, which is "
                                 f"an invalid cell configuration.")
            edge_ops[edge_name] = OP_NAMES.index(op)
        out_node += 1

    return tuple(edge_ops[k] for k in EDGE_LIST)

def convert_nb201_str_to_naslib_obj(nb201_str, naslib_obj, change_search_space=False):
    """ Given a string describing a unique architecture in the format of NASBench-201, changes the given NASLib Object,
    expected to be a NASB201HPOSearchSpace object, to reflect that architecture. Note that this function assumes that
    the other dynamic architecture parameters of the object's config have already been properly set. Alternatively,
    setting change_search_space to True will instead cause this function to modify the ConfigurationSpace of naslib_obj
    such that the cell ops are fixed to the appropriate values for all future samples drawn from the space. This still
    does not touch the other parameters of the space. Ideally, this setting should be used on the entire
    NASB201HPOSearchSpace class. """

    op_inds = convert_str_to_indices(nb201_str)

    if change_search_space:
        original_space = naslib_obj
        config_space = original_space.config_space
        known_params = {p.name: p for p in config_space.get_hyperparameters()}
        new_params = {f"Op{i}": v for i, v in enumerate(op_inds, start=1)}
        for k, v in new_params.items():
            meta = dict(known_params[k].meta, **dict(constant_overwrite=True))
            known_params[k] = ConfigSpace.Constant(k, v, meta=meta)
        new_config_space = ConfigSpace.ConfigurationSpace(f"{config_space.name}_fixed_ops")
        new_config_space.add_hyperparameters(known_params.values())

        original_space.config_space = new_config_space
    else:
        naslib_obj.clear()
        config = naslib_obj.config_space.get_default_configuration()
        config.keys()
        op_config = {f"Op{i}": op for i, op in enumerate(op_inds, start=1)}
        for k, v in op_config.items():
            config[k] = v

        naslib_obj.config = config
        naslib_obj._construct_graph()
