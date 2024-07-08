# Capillary_Homogenization

# README

## Introduction
This repository can be used to perform the homogenization approach on a coupled tissue-capillary problem, to compute the pressure (and in some case the velocity) of the blood flow.

There are 3 test cases:
- Synthetic network : a synthetic network embedded in a cubic tissue domain.
- Tumor cubic network : extraction of a cubic network from a given network of a mouse brain tumor, obtained from the REANIMATE project.
- Tumor comprehensive network : analysis on the entire tumor network. 

## Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `pandas`, `dolfin`,`fenics`,`networkx`, `petsc4py`, `block`, `xii`, `matplotlib`

## Code Structure
The code is divided in 5 files:

- `auxiliary_functions.py`: This Python file groups all the functions used to infer and analyze this problem.
- `Synthetic_network_analysis.ipynb`: Jupiter notebook that computes the pressure using the homogenization approach of the synthetic network.
- `MIXED_Synthetic_network_analysis.ipynb`: Jupiter notebook that computes the velocity and pressure using the homogenization approach of the synthetic network, using the mixed formulation.
- `Tumor_cubic_network_analysis.ipynb`: Jupiter notebook that computes the pressure using the homogenization approach of the cubic network extracted from the tumor.
- `Tumor_total_network_analysis.ipynb`: Jupiter notebook that computes the pressure using the homogenization approach of the entire tumor network.


The meshes are stored in the following folders:
- `888_mesh`: synthetic mesh.
- `Tumor_mesh_cube`: cubic tumor mesh.
- `Tumor_mesh_total`: entire tumor mesh.










