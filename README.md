# Examples

This repository is dedicated to providing examples of how to use the OAAX runtime and conversion toolchain.
The examples are provided is self-contained folders, each containing a README file that explains how to build and run
the example.

Please note that all examples can be used with any OAAX-compliant runtime. Hence, to run a model on different AI
accelerator, you only need to replace the runtime library with the one that is compatible with the accelerator.

## Structure

The repository is structured as follows:

- [c-example](c-example): Contains an example of how to use the OAAX runtime in a C program.
- [python-example](python-example): Contains an example of how to use the OAAX runtime in a Python program.
- [parallel-models](parallel-models): Contains an example of how to use the OAAX runtime to run multiple models in
  parallel.
- [image-classification-example](image-classification-example): Contains an example of how to exchange an OAAX runtime with another

## Getting started

The examples provided assume that the optimized model file has been generated using a compatible conversion toolchain.
> All conversion toolchains can be found in the [contributions](https://github.com/oaax-standard/contributions)
> repository.

For example, to optimize an ONNX model using the reference conversion toolchain, you can run the following commands:

```bash
wget -c https://download.sclbl.net/OAAX/toolchains/conversion-toolchain-latest.tar
docker load -i conversion-toolchain-latest.tar
docker run -v $(pwd):/run -it conversion-toolchain:latest /run/model.onnx /run/
```

The last command takes a model file available at `$(pwd)/model.onnx` and
generates an optimized model located in the `/run/` folder.    
The generated model is compatible for use by only the OAAX reference runtime.

> You can find other contributed conversion toolchains in
> the [contributions](https://github.com/oaax-standard/contributions) repository, in addition to documentation on how to
> use each one of them.