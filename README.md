# Examples

[![Check C/C++ Code Style](https://github.com/OAAX-standard/examples/actions/workflows/run-linter.yml/badge.svg)](https://github.com/OAAX-standard/examples/actions/workflows/run-linter.yml)
[![Run tests](https://github.com/OAAX-standard/examples/actions/workflows/run-tests.yml/badge.svg)](https://github.com/OAAX-standard/examples/actions/workflows/run-tests.yml)

A standalone example of how to use an [OAAX 2.0](https://oaax.org) runtime from an application.

It has no dependencies on other OAAX repositories or git submodules: the OAAX 2.0 interface header, a sample model and
ready-to-use CPU runtimes are all included.

## Structure

- [c-example](c-example): loads an OAAX runtime in a C program, runs a face detection model on an image and prints the
  detections. Works on Linux (x86_64, aarch64) and Windows (x86_64).

## Getting started

```bash
git clone https://github.com/OAAX-standard/examples.git
cd examples
sudo apt-get install cmake build-essential libjpeg-dev  # Ubuntu
bash c-example/build-run.sh
```

See the [c-example README](c-example/README.md) for details, and for how to use your own runtime, model and image.

## Using another accelerator

Examples work with any OAAX-compliant runtime. To run a model on a different AI accelerator, optimize the model with
that accelerator's conversion toolchain and replace the runtime library with the one built for it.

## Contributing

If you would like to contribute to this repository, please follow the guidelines in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
