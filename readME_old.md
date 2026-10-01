# [Optimizing the Cross-Sections of Lattice Structures Using Variational Autoencoders](https://ssrn.com/abstract=6128827)


**Akshay Kumar**, **Saketh Sridhara**, **Krishnan Suresh**  
Department of Mechanical Engineering, University of Wisconsin–Madison

This repository accompanies the paper:

> *[Optimizing the Cross-Sections of Lattice Structures Using Variational Autoencoders](https://ssrn.com/abstract=6128827)*

and provides a complete, end-to-end workflow for the design, optimization, and reconstruction of lattice structures with free-form, manufacturable beam cross-sections.

![Graphical Abstract](Images/GraphicalAbstract.png)

---

## Overview

This work presents a data-driven framework for optimizing lattice beam cross-sections under linear elastic loading. Beam cross-sections are parameterized using B-spline curves and optimized in a learned latent space obtained via a variational autoencoder (VAE).

Direct optimization in the B-spline space is challenging due to high dimensionality and geometric validity constraints. To address this, a VAE is trained on datasets of both valid and invalid cross-section shapes, enabling it to implicitly learn valid geometries. The resulting latent space is smooth, low-dimensional, and well-suited for gradient-based optimization.

Optimized latent variables are decoded back into B-spline representations and used to generate watertight STL files for additive manufacturing, enabling a fully digital design-to-production workflow.

---

## Workflow

The framework consists of two primary stages:

---

### 1. Data Generation and VAE Training

Use `trainVAE.py` to generate cross-section datasets and train the variational autoencoder.

This script allows you to:
- Define B-spline geometry parameters:
  - Number of control points (`n_cp`)
  - Spline degree (`k`)
- Generate datasets containing both valid and invalid cross-section shapes
- Specify VAE architecture and training parameters
- Train a VAE that maps B-spline control points and cross-section properties to a compact latent space

**Output:**  
A trained VAE model whose decoder produces valid, manufacturable B-spline cross-sections.

---

### 2. Lattice Optimization Using a Trained VAE

Latent-space optimization is performed using `optBeam.ipynb`.

This notebook allows you to:
- Load a previously trained VAE model
- Define structural optimization problems:
  - Single beams
  - Three beams
  - Lattice structures
  - More
- Specify boundary conditions, loading, and volume constraints
- Perform gradient-based optimization directly in the latent space
- Decode optimized latent variables into B-spline cross-sections
- Automatically generate watertight STL files suitable for 3D printing

Example problems are provided and can be easily extended to custom geometries and loading conditions.

---

## Key Features

- Free-form beam cross-section representation using B-splines
- Data-driven enforcement of geometric validity via a VAE
- Low-dimensional latent space for efficient optimization
- Differentiable decoder enabling automatic differentiation
- Support for solid and hollow cross-sections
- Automated generation of watertight STL files for additive manufacturing

---

## Installation


### 1. Create a Conda Environment

```bash
conda create -n LatticeCSOpt python=3.10
conda activate LatticeCSOpt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
See `requirements.txt` for the complete list of dependencies.

---

## Citation

If you use this code in your research, please cite the associated paper.
(Citation information will be added upon publication.)

---

## License

This project currently has no license.
All rights reserved.

---

## Acknowledgments

This work was supported by the U.S. Office of Naval Research under the PANTHER award
**N00014-21-1-2916**, monitored by **Dr. Timothy Bentley**.

