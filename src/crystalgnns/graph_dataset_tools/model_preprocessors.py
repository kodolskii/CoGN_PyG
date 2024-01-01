import numpy as np
# import tensorflow as tf
import torch
from typing import Tuple, List
# from tensorflow import (
#     TensorSpec,
#     convert_to_tensor,
#     float64,
#     float32,
#     int64,
#     int32,
#     RaggedTensor,
#     Tensor,
# )
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from crystalgnns.graph_dataset_tools.graph_tuple import GraphTuple
from kgcnn.graph.adj import get_angle_indices
from enum import Enum
from crystalgnns.graph_dataset_tools.line_graph import get_line_graph


class ModelPreprocessor:
    """Abstract class for classes that converts GraphTuples to Tensors."""

    def __init__(self, output_signature=None, input_signature=None):
        self.output_signature = output_signature
        self.input_signature = input_signature

    @property
    def signature(self):
        """Returns signature of model input and output."""
        return self.input_signature, self.output_signature

    def to_tensor(self, graph_tuple: GraphTuple):
        """Generate Tensors from a singleton GraphTuple.

        Args:
            graph_tuple (GraphTuple): Singleton (list of one item) GraphTuple to create Tensors for. 

        Returns:
            Tensors for the given GraphTuple singleton.
        """
        raise NotImplementedError()

    def get_dataset(
        self,
        graph_tuple: GraphTuple,
        indices=None,
        batch_size=32,
        cache_file=None,
        remove_no_edge_graphs=False,
    ):
        """Generate a TensorFlow Dataset from the 

        Args:
            graph_tuple (GraphTuple): GraphTuple dataset to convert to a Tensorflow dataset.
            indices (list, optional): Which graph instances from the GraphTuple dataset to include in the TF dataset.
                None inlcudes all graph instances.
                Defaults to None.
            batch_size (int, optional): Batch size of the TF dataset. Defaults to 32.
            cache_file (optional): Cache file of the TF dataset. Defaults to None.
            remove_no_edge_graphs (bool, optional): Whether to remove graphs with no edges.
                Defaults to False.

        Returns:
            tf.Dataset: Tensorflow Dataset.
        """

        if indices is None:
            indices = range(len(graph_tuple))

        def dataset_generator():
            for i in indices:
                if not remove_no_edge_graphs or graph_tuple[i].num_edges[0] > 0:
                    yield self.to_tensor(graph_tuple[i])

        # dataset = tf.data.Dataset.from_generator(
        #     dataset_generator, output_signature=self.signature
        # ).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
        dataset = Dataloader(dataset_generator, batch_size = batch_size)

        if cache_file is not None:
            dataset = dataset.cache(cache_file)

        return dataset


class GNPreprocessor(ModelPreprocessor):
    """Preprocessor, with predefined edge, node and graph properties to convert GraphTuples to Tensors."""

    class Inputs(Enum):
        """Predefined edge, node and graph properties for crystal graphs.

        This are the properties available to use with the GNPreprocessor."""

        # distance = TensorSpec(shape=(None,), dtype=float32, name="distance")
        # offset = TensorSpec(shape=(None, 3), dtype=float32, name="offset")
        # cell_translation = TensorSpec(
        #     shape=(None, 3), dtype=float32, name="cell_translations"
        # )
        # symmop = TensorSpec(shape=(None, 4, 4), dtype=float32, name="symmops")
        # voronoi_ridge_area = TensorSpec(
        #     shape=(None,), dtype=float32, name="voronoi_ridge_area"
        # )
        # atomic_number = TensorSpec(shape=(None,), dtype=int32, name="atomic_number")
        # frac_coords = TensorSpec(shape=(None, 3), dtype=float32, name="frac_coords")
        # coords = TensorSpec(shape=(None, 3), dtype=float32, name="coords")
        # multiplicity = TensorSpec(shape=(None,), dtype=int32, name="multiplicity")
        # line_graph_edge_indices = TensorSpec(
        #     shape=(None, 2), dtype=int32, name="line_graph_edge_indices"
        # )
        # edge_indices = TensorSpec(shape=(None, 2), dtype=int32, name="edge_indices")
        # lattice_matrix = TensorSpec(shape=(3, 3), dtype=float32, name="lattice_matrix")

    #PyG:

            distance = torch.empty([0,], dtype=torch.float32)
        offset = torch.empty([0,3], dtype=torch.float32)
        cell_translation = torch.empty([0,3], dtype=torch.float32)
        symmop = torch.empty([0,4,4], dtype=torch.float32)
        voronoi_ridge_area = torch.empty([0,], dtype=torch.float32)
        atomic_number = torch.empty([0,], dtype=torch.float32)
        frac_coords = torch.empty([0,3], dtype=torch.float32)
        coords = torch.empty([0,3], dtype=torch.float32)
        multiplicity = torch.empty([0,], dtype=torch.float32)
        line_graph_edge_indices = torch.empty([0,2], dtype=torch.float32)
        edge_indices = torch.empty([0,2], dtype=torch.float32)
        lattice_matrix = torch.empty([3,3], dtype=torch.float32)

    def __init__(
        self,
        output_signature=None,
        inputs=[
            Inputs.distance,
            Inputs.offset,
            Inputs.atomic_number,
            Inputs.multiplicity,
            Inputs.line_graph_edge_indices,
            Inputs.edge_indices,
        ],
        line_graph_directions=[1, 1],
    ):
        """Initialized a preprocessor, with predefined edge, node and graph properties to convert GraphTuples to Tensors

        Args:
            output_signature (_type_, optional): Signature to build a TensorFlow Dataset an iterator. Defaults to None.
            inputs (list, optional): Specifies the model inputs (in the right order),
                so that GraphTuples can be converted to tensors that suit the model. Defaults to [Inputs.distance, Inputs.offset, Inputs.atomic_number, Inputs.multiplicity, Inputs.line_graph_edge_indices, Inputs.edge_indices].
            line_graph_directions (list, optional): Line graph variant specification. Defaults to [1,1].
        """
        self.inputs = inputs
        self.line_graph_directions = line_graph_directions
        input_signature = tuple([i.value for i in inputs])
        if output_signature is None:
            # output_signature = TensorSpec(shape=(), dtype=float64)
            output_signature = torch.empty([0,3], dtype=torch.float64)
        super().__init__(output_signature, input_signature)

    def to_nested_tensor(self, graph_tuple: GraphTuple) -> Tuple[List, Tensor]:
        """Builds RaggedTensors for the whole GraphTuple dataset.

        Args:
            graph_tuple (GraphTuple): Dataset to convert to RaggedTensors

        Returns:
            Tuple[List, Tensor]: Tuple of input tensors and an output tensor (target values).
        """
        inputs_ = []

        for input_ in self.input_signature:
            if input_ == self.Inputs.distance.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_attributes["distance"]
                    )
                )
            if input_ == self.Inputs.offset.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_attributes["offset"]
                    )
                )
            if input_ == self.Inputs.cell_translation.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_attributes["cell_translation"]
                    )
                )
            if input_ == self.Inputs.symmop.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_attributes["symmop"]
                    )
                )
            if input_ == self.Inputs.voronoi_ridge_area.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_attributes["voronoi_ridge_area"]
                    )
                )
            if input_ == self.Inputs.atomic_number.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.node_attributes["atomic_number"],
                    )
                )
            if input_ == self.Inputs.frac_coords.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.node_attributes["frac_coords"]
                    )
                )
            if input_ == self.Inputs.coords.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.node_attributes["coords"]
                    )
                )
            if input_ == self.Inputs.multiplicity.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.node_attributes["multiplicity"]
                    )
                )
            if input_ == self.Inputs.line_graph_edge_indices.value:
                line_graph_edge_indices = []
                for g in graph_tuple:
                    line_graph_edge_indices.append(
                        get_line_graph(
                            g.edge_indices[:], directions=self.line_graph_directions
                        )
                    )
                row_lengths = [len(l) for l in line_graph_edge_indices]
                line_graph_edge_indices_ragged = torch.nested.nested_tensor(
                    np.concatenate(line_graph_edge_indices)
                )
                inputs_.append(line_graph_edge_indices_ragged)
            if input_ == self.Inputs.edge_indices.value:
                inputs_.append(
                    torch.nested.nested_tensor(
                        graph_tuple.edge_indices[:][:, [1, 0]]
                    )
                )
            if input_ == self.Inputs.lattice_matrix.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.graph_attributes["lattice_matrix"],dtype =  input_.dtype
                    )
                )

        target = torch.tensor(
            graph_tuple.graph_attributes["label"],dtype =   self.output_signature.dtype
        )
        return tuple(inputs_), target

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        inputs_ = []

        for input_ in self.input_signature:
            if input_ == self.Inputs.distance.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.edge_attributes["distance"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.offset.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.edge_attributes["offset"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.cell_translation.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.edge_attributes["cell_translation"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.symmop.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.edge_attributes["symmop"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.voronoi_ridge_area.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.edge_attributes["voronoi_ridge_area"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.atomic_number.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.node_attributes["atomic_number"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.frac_coords.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.node_attributes["frac_coords"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.coords.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.node_attributes["coords"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.multiplicity.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.node_attributes["multiplicity"],dtype =   input_.dtype
                    )
                )
            if input_ == self.Inputs.line_graph_edge_indices.value:
                line_graph_edge_indices = get_line_graph(
                    graph_tuple.edge_indices[:], directions=self.line_graph_directions
                )
                line_graph_edge_indices = line_graph_edge_indices[:, [1, 0]]
                inputs_.append(torch.tensor(line_graph_edge_indices,dtype =    input_.dtype))
            if input_ == self.Inputs.edge_indices.value:
                inputs_.append(
                    torch.tensor(graph_tuple.edge_indices[:, [1, 0]],dtype =    input_.dtype)
                )
            if input_ == self.Inputs.lattice_matrix.value:
                inputs_.append(
                    torch.tensor(
                        graph_tuple.graph_attributes["lattice_matrix"][0],dtype =    input_.dtype
                    )
                )

        target = torch.tensor(
            graph_tuple.graph_attributes["label"][0], dtype =   self.output_signature.dtype
        )
        return tuple(inputs_), target


class SchNetPreprocessor(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):

        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64, name="atomic_numbers"),
                TensorSpec(shape=(None, 1), dtype=float64, name="edge_distances"),
                TensorSpec(shape=(None, 2), dtype=int64, name="edge_indices"),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = convert_to_tensor(graph_tuple.edge_indices[:, [1, 0]], int64)
        atomic_numbers = convert_to_tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        edge_distances = convert_to_tensor(
            np.expand_dims(graph_tuple.edge_attributes["distance"], -1),
            self.input_signature[1].dtype,
        )

        target = convert_to_tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (node_attributes, edge_distances, edge_indices), target


class CGCNNPreprocessorEdgeDistances(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):
        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64, name="atomic_numbers"),
                TensorSpec(shape=(None,), dtype=float64, name="edge_distances"),
                TensorSpec(shape=(None, 2), dtype=int64, name="edge_indices"),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = convert_to_tensor(graph_tuple.edge_indices[:, [1, 0]], int64)
        atomic_numbers = convert_to_tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        edge_distances = convert_to_tensor(
            graph_tuple.edge_attributes["distance"], self.input_signature[1].dtype
        )
        target = convert_to_tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (node_attributes, edge_distances, edge_indices), target


class CGCNNPreprocessorSuper(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):
        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64, name="atomic_numbers"),
                TensorSpec(shape=(None, 3), dtype=float64, name="frac_coords"),
                TensorSpec(shape=(3, 3), dtype=float64, name="lattice_matrix"),
                TensorSpec(shape=(None, 2), dtype=int64, name="edge_indices"),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = convert_to_tensor(graph_tuple.edge_indices[:, [1, 0]], int64)
        atomic_numbers = convert_to_tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        frac_coords = convert_to_tensor(
            graph_tuple.node_attributes["frac_coords"], self.input_signature[1].dtype
        )
        lattice_matrix = convert_to_tensor(
            graph_tuple.graph_attributes["lattice_matrix"][0],
            self.input_signature[2].dtype,
        )

        target = convert_to_tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (node_attributes, frac_coords, lattice_matrix, edge_indices), target


class CGCNNPreprocessorUnit(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):
        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64, name="atom_attributes"),
                TensorSpec(shape=(None, 3), dtype=float64, name="frac_coords"),
                TensorSpec(shape=(None, 3), dtype=float64, name="cell_translations"),
                TensorSpec(shape=(3, 3), dtype=float64, name="lattice_matrix"),
                TensorSpec(shape=(None, 2), dtype=int64, name="edge_indices"),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = convert_to_tensor(graph_tuple.edge_indices[:, [1, 0]], int64)
        atomic_numbers = convert_to_tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        frac_coords = convert_to_tensor(
            graph_tuple.node_attributes["frac_coords"], self.input_signature[1].dtype
        )
        cell_translations = convert_to_tensor(
            graph_tuple.edge_attributes["cell_translation"],
            self.input_signature[2].dtype,
        )
        lattice_matrix = convert_to_tensor(
            graph_tuple.graph_attributes["lattice_matrix"][0],
            self.input_signature[3].dtype,
        )

        target = convert_to_tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (
            node_attributes,
            frac_coords,
            cell_translations,
            lattice_matrix,
            edge_indices,
        ), target


class CGCNNPreprocessorASU(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):
        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64, name="atom_attributes"),
                TensorSpec(shape=(None, 3), dtype=float32, name="frac_coords"),
                TensorSpec(shape=(None,), dtype=int64, name="multiplicities"),
                TensorSpec(shape=(None, 3), dtype=float32, name="cell_translations"),
                TensorSpec(shape=(None, 4, 4), dtype=float32, name="symmops"),
                TensorSpec(shape=(3, 3), dtype=float32, name="lattice_matrix"),
                TensorSpec(shape=(None, 2), dtype=int64, name="edge_indices"),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = convert_to_tensor(graph_tuple.edge_indices[:, [1, 0]], int64)
        atomic_numbers = convert_to_tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        frac_coords = convert_to_tensor(
            graph_tuple.node_attributes["frac_coords"], self.input_signature[1].dtype
        )
        multiplicities = convert_to_tensor(
            graph_tuple.node_attributes["multiplicity"], self.input_signature[2].dtype
        )
        cell_translations = convert_to_tensor(
            graph_tuple.edge_attributes["cell_translation"],
            self.input_signature[3].dtype,
        )
        symmops = convert_to_tensor(
            graph_tuple.edge_attributes["symmop"], self.input_signature[4].dtype
        )
        lattice_matrix = convert_to_tensor(
            graph_tuple.graph_attributes["lattice_matrix"][0],
            self.input_signature[5].dtype,
        )

        target = convert_to_tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (
            node_attributes,
            frac_coords,
            multiplicities,
            cell_translations,
            symmops,
            lattice_matrix,
            edge_indices,
        ), target


class MegNetPreprocessor(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):

        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64),
                TensorSpec(shape=(None, 1), dtype=float64),
                TensorSpec(shape=(None, 2), dtype=int64),
                TensorSpec(shape=(), dtype=float64),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices = torch.tensor(
            graph_tuple.edge_indices[:, [1, 0]], self.input_signature[0].dtype
        )
        atomic_numbers = torch.tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[1].dtype
        )
        node_attributes = atomic_numbers
        edge_distances = torch.tensor(
            np.expand_dims(graph_tuple.edge_attributes["distance"], -1),
            self.input_signature[2].dtype,
        )
        graph_attributes = torch.tensor(0.0, self.input_signature[3].dtype)

        target = torch.tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (node_attributes, edge_distances, edge_indices, graph_attributes), target


class DimeNetPreprocessor(ModelPreprocessor):
    def __init__(self, output_signature=None, input_signature=None):

        super().__init__(output_signature, input_signature)

        if self.output_signature is None:
            self.output_signature = TensorSpec(shape=(), dtype=float64)
        if self.input_signature is None:
            self.input_signature = (
                TensorSpec(shape=(None,), dtype=int64),
                TensorSpec(shape=(None, 3), dtype=float64),
                TensorSpec(shape=(None, 2), dtype=int64),
                TensorSpec(shape=(None, 2), dtype=int64),
            )

    def to_tensor(self, graph_tuple: GraphTuple):
        assert graph_tuple.num_nodes.shape[0] == 1

        edge_indices =  torch.tensor(
            graph_tuple.edge_indices[:, [1, 0]], self.input_signature[2].dtype
        )
        atomic_numbers =  torch.tensor(
            graph_tuple.node_attributes["atomic_number"], self.input_signature[0].dtype
        )
        node_attributes = atomic_numbers
        edge_diff =  torch.tensor(
            graph_tuple.edge_attributes["offset"], self.input_signature[1].dtype
        )
        angle_indices =  torch.tensor(
            graph_tuple.edge_indices[:], allow_multi_edges=True
        )[2]

        target =  torch.tensor(
            graph_tuple.graph_attributes["label"][0], self.output_signature.dtype
        )

        return (node_attributes, edge_diff, edge_indices, angle_indices), target
