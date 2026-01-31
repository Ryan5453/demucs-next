"""
This module runs automatically when Python starts.
It clears LD_LIBRARY_PATH to force PyTorch to use its bundled cuDNN.
"""
import os

# Remove LD_LIBRARY_PATH to prevent the system cuDNN from being loaded
# PyTorch will then use its bundled cuDNN 9.10.2
os.environ.pop("LD_LIBRARY_PATH", None)
