import ctypes

import numpy as np
from numpy import prod


# Define the C enum
class tensor_data_type(ctypes.c_int):
    DATA_TYPE_FLOAT = 1
    DATA_TYPE_UINT8 = 2
    DATA_TYPE_INT8 = 3
    DATA_TYPE_UINT16 = 4
    DATA_TYPE_INT16 = 5
    DATA_TYPE_INT32 = 6
    DATA_TYPE_INT64 = 7
    DATA_TYPE_STRING = 8
    DATA_TYPE_BOOL = 9
    DATA_TYPE_DOUBLE = 11
    DATA_TYPE_UINT32 = 12
    DATA_TYPE_UINT64 = 13


# Define the C struct
class tensors_struct(ctypes.Structure):
    _fields_ = [
        ('num_tensors', ctypes.c_size_t),
        ('names', ctypes.POINTER(ctypes.c_char_p)),
        ('data_types', ctypes.POINTER(tensor_data_type)),
        ('ranks', ctypes.POINTER(ctypes.c_size_t)),
        ('shapes', ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t))),
        ('data', ctypes.POINTER(ctypes.c_void_p)),
    ]


_numpy_type_to_enum = {
    'float32': tensor_data_type.DATA_TYPE_FLOAT,
    'float64': tensor_data_type.DATA_TYPE_DOUBLE,
    'uint8': tensor_data_type.DATA_TYPE_UINT8,
    'int8': tensor_data_type.DATA_TYPE_INT8,
    'uint16': tensor_data_type.DATA_TYPE_UINT16,
    'int16': tensor_data_type.DATA_TYPE_INT16,
    'uint32': tensor_data_type.DATA_TYPE_UINT32,
    'int32': tensor_data_type.DATA_TYPE_INT32,
    'uint64': tensor_data_type.DATA_TYPE_UINT64,
    'int64': tensor_data_type.DATA_TYPE_INT64,
    'string': tensor_data_type.DATA_TYPE_STRING,
    'bool': tensor_data_type.DATA_TYPE_BOOL,
}

_numpy_type_to_ctype = {
    'float32': ctypes.c_float,
    'float64': ctypes.c_double,
    'uint8': ctypes.c_uint8,
    'int8': ctypes.c_int8,
    'uint16': ctypes.c_uint16,
    'int16': ctypes.c_int16,
    'uint32': ctypes.c_uint32,
    'int32': ctypes.c_int32,
    'uint64': ctypes.c_uint64,
    'int64': ctypes.c_int64,
    'bool': ctypes.c_bool,
    'string': ctypes.c_char_p,
}


def __numpy_to_c_tensor_data_type(data_type):
    """Convert Numpy data type to C data type.

    Args:
        data_type (np.dtype): Numpy data type.
    Returns:
        tensor_data_type: C data type.
    """
    return _numpy_type_to_enum[data_type.name]


def __c_tensor_data_type_to_numpy(data_type):
    """Convert C data type to Numpy data type.

    Args:
        data_type (tensor_data_type): C data type.
    Returns:
        np.dtype: Numpy data type.
    """
    for np_type, c_type in _numpy_type_to_enum.items():
        if c_type == data_type.value:
            return np.dtype(np_type)
    raise ValueError(f"Unsupported data type: {data_type}")


def __get_element_size(data_type):
    """Get the size of the data type in bytes.

    Args:
        data_type (np.dtype): Numpy data type.
    Returns:
        int: Size of the data type in bytes.
    """
    return np.dtype(data_type).itemsize


def __get_ctypes_type(data_type):
    """Get the ctypes type for the data type.

    Args:
        data_type (np.dtype): Numpy data type.
    Returns:
        ctypes type: Ctypes data type.
    """
    return _numpy_type_to_ctype[np.dtype(data_type).name]


# Function to build input tensors
def numpy_to_c_struct(tensor_data):
    """Build input tensors for the C runtime.

    Args:
        tensor_data (Dict[str, np.ndarray]): Dictionary containing the tensor name and data.
    Returns:
        tensors_struct: Input tensors for the C runtime.
    """
    num_tensors = len(tensor_data)

    # Allocate memory for the tensors_struct
    tensors = tensors_struct()
    tensors.num_tensors = num_tensors

    # Allocate memory for names, data_types, ranks, shapes, and data
    tensors.names = (ctypes.c_char_p * num_tensors)()
    tensors.data_types = (tensor_data_type * num_tensors)()
    tensors.ranks = (ctypes.c_size_t * num_tensors)()
    tensors.shapes = (ctypes.POINTER(ctypes.c_size_t) * num_tensors)()
    tensors.data = (ctypes.c_void_p * num_tensors)()

    # Populate the tensors
    for i, tensor_name in enumerate(tensor_data):
        np_array = tensor_data[tensor_name]
        shape = np_array.shape
        data = np_array.flatten().tobytes()

        tensors.names[i] = tensor_name.encode('utf-8')
        tensors.data_types[i] = __numpy_to_c_tensor_data_type(np_array.dtype)
        tensors.ranks[i] = np_array.ndim
        tensors.shapes[i] = (ctypes.c_size_t * len(shape))()
        for j, dim in enumerate(shape):
            tensors.shapes[i][j] = dim

        # Allocate memory for the data
        data_pointer = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p)
        tensors.data[i] = data_pointer

    return tensors


def c_struct_to_numpy(output_tensors):
    """Parse output tensors from the C runtime.

    Args:
        output_tensors (tensors_struct): Output tensors from the C runtime.
    Returns:
        Dict[str, np.ndarray]: Dictionary containing the tensor name and data.
    """
    tensor_data = {}
    for i in range(output_tensors.num_tensors):
        tensor_name = output_tensors.names[i].decode('utf-8')
        data_type = __c_tensor_data_type_to_numpy(output_tensors.data_types[i])
        rank = output_tensors.ranks[i]
        shape = [output_tensors.shapes[i][j] for j in range(rank)]
        tensor_size = prod(shape)
        ctype = __get_ctypes_type(data_type)
        data = ctypes.cast(output_tensors.data[i], ctypes.POINTER(ctype * tensor_size))

        # Convert the data to a Numpy array
        np_array = np.frombuffer(data.contents, dtype=data_type)
        np_array = np_array.reshape(shape)

        tensor_data[tensor_name] = np_array
    return tensor_data


if __name__ == '__main__':
    # Example usage
    tensor_data = {
        'tensor1': np.random.randint(0, 10, (1, 11)).astype('float32'),
        'tensor2': np.random.randint(-10, 0, (20, 1)).astype('int8'),
        'tensor3': np.random.rand(5, 5).astype('float64'),
        'tensor4': np.random.randint(0, 2, (1, 2, 3, 4)).astype('bool'),
    }

    c_tensors = numpy_to_c_struct(tensor_data)
    tensor_data2 = c_struct_to_numpy(c_tensors)

    # Compare tensor_data and tensor_data2
    for tensor_name in tensor_data:
        np.testing.assert_array_equal(tensor_data[tensor_name], tensor_data2[tensor_name])
    print('All arrays are equal!')
