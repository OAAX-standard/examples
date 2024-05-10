# C example

This example demonstrates the usage of the OAX runtime library to load an optimized model and run it on CPU.

## Overview

This folder contains the source code of a C program that will load the runtime library, load the model, and act like a
server that can take messages from any other process through shared memory and send back the results to the client.
Those message need to be in a specific format so that the server can parse them correctly to extract the tensor data.

The `artifacts/` directory contains a Python script that serves as a client to the server. It sends a message to the
server
containing the input tensor data, and the server will send back the output tensor data.

The server's source code is available in the `src/` directory.

## Requirements

This example is expected to run on an Ubuntu 22.04 LTS machine with an X86_64 architecture, with Python 3.8
installed.   
Additionally, to compile the server, you need a cross-compilation toolchain that can be downloaded
from [here](https://download.sclbl.net/toolchains/x86_64-unknown-linux-gnu-gcc-9.5.0.tar.gz), and extracted to
the `/opt` directory.

Moreover, you need to pick an OAX runtime and conversion toolchain that can run on the target architecture. You can use
the X86_64 runtime and conversion toolchain provided in
the [contributions](https://github.com/oax-standard/contributions/tree/develop/X86_64), then place them in
the `artifacts/` directory.

## Getting started

1. The first step to deploy the ONNX model is optimize it for CPU. That can be achieved by running the conversion
   toolchain using:

```bash
bash convert-model.sh
```

This will generate an optimized model `model-optimized.onnx` that can be used by the runtime.

2. To build the server, run the following command:

```bash
bash build-server.sh
```

3. Once the server is built, you have to start the client first. To do so, run the following command:

```bash
python3 artifacts/client.py
```

The client will print certain parameters that need to be provided to the server in stdout.
These parameters are used for IPC purposes.
An example of the parameters is as follows

```
Engine paramters (engine_pipe_name, module_pipe_name, shm_id, shm_key): artifacts/engine_pipe artifacts/module_pipe 3047430 17104897
```

4. Start the server by running the following command:

```bash
./c_example <runtime library path> <model filepath> <server pipe> <client pipe> <shm id> <shm key>
```

Where:

- `<runtime library path>` is the path to the runtime library shared object file.
- `<model filepath>` is the path to the optimized model file.
- `<server pipe>` is the path to the server pipe.
- `<client pipe>` is the path to the client pipe.
- `<shm id>` is the shared memory ID.
- `<shm key>` is the shared memory key.

The last 4 parameters are provided when the client is started.

For example:

```bash
cd build
./c_example ../artifacts/libRuntimeLibrary.so ../artifacts/model-simplified.onnx ...
```

5. The server will start and wait for the client to send a message. Once the client sends a message, the server will
   parse it, run the model, and send back the results to the client. The client will then visualize the faces on the
   image and save the result in the `artifacts/` directory. For the sake of simplicity, the client will keep sending the
   same message (since it's using the same JPG image). But that can be easily improved to send different messages.