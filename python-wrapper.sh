#!/bin/bash
# Clear LD_LIBRARY_PATH to force PyTorch to use its bundled cuDNN
unset LD_LIBRARY_PATH
exec python "$@"
