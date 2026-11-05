#!/bin/bash
# timm 0.3.2 imports torch._six.container_abcs, removed in newer PyTorch.
set -e
F=$CONDA_PREFIX/lib/python3.9/site-packages/timm/models/layers/helpers.py
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' "$F"
grep -n "container_abcs" "$F"
