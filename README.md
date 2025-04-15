# Examples

This repository is dedicated to providing examples of how to use the OAAX runtime and conversion toolchain.
The examples are provided is self-contained folders, each containing a README file that explains how to build and run
the example.

Please note that examples can be used with any OAAX-compliant runtime. Hence, to run a model on different AI
accelerator, you need to compile the ONNX model using the right conversion toolchain and replace the runtime library with the one that is compatible with the accelerator.

## Structure

The repository is structured as follows:

- [c-example](c-example): Contains an example of how to use an OAAX runtime in a C program.
- [tools](tools): Contains a set of utility tools that can be used in AI applications for benchmarking, profiling, and debugging runtimes.

## Getting started

> Note: You can find an sample ONNX model in [c-example/artifacts/model.onnx](c-example/artifacts/model.onnx).

### Compile the model for the target hardware

The examples provided assume that the optimized model file has been generated using a compatible conversion toolchain.
> All conversion toolchains can be found in the [contributions](https://github.com/oaax-standard/contributions)
> repository.

For instance, to optimize an ONNX model for the CPU, download the toolchain from [Google Drive](https://drive.google.com/file/d/1Xz9m1ATwmM9w81bDuZ1ibNqcyLmOKFWW/view) and run the following commands:

```bash
docker load -i conversion-toolchain-latest.tar
docker run --rm -v $(pwd):/run -it conversion-toolchain:latest /run/model.onnx /run/
```

The last command takes a model file available at `$(pwd)/model.onnx` and
generates an optimized model that will be saved in the `/run/` folder.    
The generated model `model-simplified.onnx` is compatible for use only by the OAAX CPU runtime.

> You can find other contributed conversion toolchains in
> the [contributions](https://github.com/oaax-standard/contributions) repository, in addition to documentation on how to
> use each one of them.


### Download the runtime

Depending on the target hardware, you need to download the appropriate runtime. The runtimes are available in the [contributions](https://github.com/oaax-standard/contributions) repository.

For example, to use the CPU runtime on an x86_64 machine, run the following command:

```bash
wget https://artifactory.nxvms.dev/artifactory/nxai_open/OAAX/runtimes/async/cpu-x86_64-ort.tar.gz
tar -xvf cpu-x86_64-ort.tar.gz # extract the runtime library: libRuntimeLibrary.so
```

### Clone the repository

Clone the repository and its submodules then navigate to the `examples` folder:

```bash
git clone --recurse-submodules https://github.com/OAAX-standard/examples.git
cd examples
```

### Build the example

Currently, there is only one example available, which is a C example. To build and use the example, please refer to the [README](c-example/README.md) file in the `c-example` folder for detailed instructions.


## Contributing

If you would like to contribute to this repository, please follow the guidelines in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.