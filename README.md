# TorchSignal
a module doing some DSP on torch tensors for HPC

This is, by far, not perfect!

The main functions here are 1D convolutions and window wrapping.
Every now and then, functions will be added and the module will get its own folder structure.

Right now it is mostly a byproduct of my resent Bayes by Backprop implementation, which does include some 1D DSP layers.

## Requirements

Python (2.7+, 3.6+)
-------------
PyTorch (latest)
Numpy (latest)
math

(Particular versioning might come later!)

## Bugs

The "in place" convolution just stops before swapping over instead of embodying the valid convolution method.

## Alternatives

There are preimplemented functions, like fft and ifft, which can make life easier since:
$f[n]\timesg[n] = F[k]G[k]$
