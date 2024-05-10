import ctypes
import os
import queue
import threading
from typing import Dict

import numpy as np

from runtime_utils import tensors_struct, numpy_to_c_struct, c_struct_to_numpy


class OAXRuntime:

    def __init__(self, library_path: str):
        selfimg = ctypes.CDLL('')
        selfimg.dlmopen.restype = ctypes.c_void_p
        selfimg.dlerror.restype = ctypes.c_char_p
        dlopen_flags = os.RTLD_NOW | os.RTLD_DEEPBIND | os.RTLD_GLOBAL

        # Open the library in a separate namespace using dlmopen
        _h = selfimg.dlmopen(-1, library_path.encode('utf-8'), dlopen_flags)
        lib = ctypes.CDLL(library_path, handle=_h, mode=dlopen_flags)

        # Define the function signatures
        lib.runtime_initialization.argtypes = []
        lib.runtime_initialization.restype = ctypes.c_int

        lib.runtime_model_loading.argtypes = [ctypes.c_char_p]
        lib.runtime_model_loading.restype = ctypes.c_int

        lib.runtime_inference_execution.argtypes = [ctypes.POINTER(tensors_struct), ctypes.POINTER(tensors_struct)]
        lib.runtime_inference_execution.restype = ctypes.c_int

        lib.runtime_inference_cleanup.argtypes = []
        lib.runtime_inference_cleanup.restype = ctypes.c_int

        lib.runtime_destruction.argtypes = []
        lib.runtime_destruction.restype = ctypes.c_int

        lib.runtime_error_message.argtypes = []
        lib.runtime_error_message.restype = ctypes.c_char_p

        lib.runtime_version.argtypes = []
        lib.runtime_version.restype = ctypes.c_char_p

        lib.runtime_name.argtypes = []
        lib.runtime_name.restype = ctypes.c_char_p

        self.lib = lib

    @property
    def name(self):
        runtime_name = self.lib.runtime_name()
        runtime_name = ctypes.cast(runtime_name, ctypes.c_char_p).value.decode('utf-8')
        return runtime_name

    @property
    def version(self):
        runtime_version = self.lib.runtime_version()
        runtime_version = ctypes.cast(runtime_version, ctypes.c_char_p).value.decode('utf-8')
        return runtime_version

    def error_message(self):
        return self.lib.runtime_error_message()

    def initialize(self):
        exit_code = self.lib.runtime_initialization()

    def load_model(self, model_path: str):
        model_path = bytes(model_path, 'utf-8')
        exit_code = self.lib.runtime_model_loading(model_path)

    def inference(self, input_data: Dict[str, np.ndarray]):
        input_tensors = numpy_to_c_struct(input_data)
        output_tensors = tensors_struct()
        exit_code = self.lib.runtime_inference_execution(input_tensors, output_tensors)
        output_data = c_struct_to_numpy(output_tensors)
        return output_data

    def inference_cleanup(self):
        exit_code = self.lib.runtime_inference_cleanup()

    def destroy(self):
        exit_code = self.lib.runtime_destruction()


class ThreadedRuntime(OAXRuntime):
    def __init__(self, library_path: str):
        super().__init__(library_path)

        self.task_queue = queue.Queue()  # Initialize the task queue
        self.results = queue.Queue()  # Initialize the results queue
        self.worker_thread = threading.Thread(target=self._process_tasks)  # Initialize the worker thread
        self.worker_thread.start()  # Start the worker thread

    def inference(self, input_data: Dict[str, np.ndarray]):
        self._add_task('inference', input_data)

    def inference_cleanup(self):
        self._add_task('inference_cleanup')

    def destroy(self):
        self._add_task('destroy')

    def _process_tasks(self):
        """
        Continuously processes tasks from the task queue.
        """
        while True:
            task = self.task_queue.get()  # Get the next task from the queue
            if task is None:  # If the task is None, it's a signal to stop the thread
                break
            try:
                method, args, kwargs = task
                res = method(*args, **kwargs)
                # Put the result in the results queue if the function has a return value
                if _does_function_have_return(method):
                    self.results.put(res)
            except Exception as e:
                print(f"Error running task in thread: {e}")
            finally:
                self.task_queue.task_done()  # Mark the task as done

    def _add_task(self, method_name: str, *args, **kwargs):
        """
        Adds a task to the task queue.
        """
        try:
            method = getattr(super(), method_name)
            self.task_queue.put((method, args, kwargs))
        except AttributeError:
            print(f"Method '{method_name}' not found in class '{self.__class__.__name__}'")

    def get_last_result(self, timeout=None):
        """ Gets the last result from the results queue.

        Args:
            timeout (float): Timeout in seconds.

        Returns:
            Any: Result from the results queue.
        """
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            print(f"Timeout after {timeout}. No results in the queue.")
            return None

    def stop_thread(self):
        """
        Signals the worker thread to stop.
        """
        self.task_queue.put(None)  # Add a None task to signal the thread to stop
        self.worker_thread.join()  # Wait for the thread to finish


def _does_function_have_return(func):
    from inspect import getsourcelines

    lines, _ = getsourcelines(func)
    return any("return" in line for line in lines)  # might give false positives, use regex for better checks
