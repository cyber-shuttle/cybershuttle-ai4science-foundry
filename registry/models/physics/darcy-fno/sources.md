# Sources and Provenance

## Model Source
Fourier Neural Operator (FNO)
Repository: https://github.com/neuraloperator/neuraloperator
Validated commit: 00b7d86f8d74ff0af55da53eb585fe26df9c71f0
Model file: neuraloperator/neuralop/models/fno.py  
License: MIT

## Dataset Source
Darcy dataset included with NeuralOperator  
Path: neuraloperator/neuralop/data/datasets/data/darcy_test_16.pt  
Contains:
- x: boolean permeability fields
- y: pressure solutions

## Environment Source
Environment defined in `environment.yml`  
NeuralOperator installed via:
`pip install -e ../neuraloperator`

## SLURM Execution Environment
Account: coc  
Partition: coc-cpu  
Node: atl1-1-02-010-2-1  