# Welcome to *ARPIST* Project #

## Introduction ##

Welcome to the `ARPIST` package! `ARPIST` stands for `A`nchored `R`adially `P`rojected with `I`ntegration on `S`pherical `T`riangles, which is a quadrature rule generator for accurate and stable integration of functions on spherical triangles. `ARPIST` is based on an efficient and easy-to-implement transformation to the spherical triangle from its corresponding linear triangle via radial projection to achieve high accuracy and efficiency.

## Installation ##

To download the latest version of the code, use the command

```console
git clone https://github.com/numgeom/ARPIST.git
```
Use `git pull` to download any new changes added since `git clone` or last `git pull`. Alternatively, use `git checkout v[GLOBAL].[MAJOR].[MINOR]` to download a specific version.

## Usage ##

We provide the matlab and python implementation and three coarse test meshes for the demo scripts. 

For the matlab implementation, please see the `test_integration_over_whole_sphere.m` for example. 
We use `compute_sphere_quadrature` to generate quadrature points and corresponding weights on the sphere, which could be reused for the integration of different functions.
We also provide another implementation `spherical_integration` for integration over some spherical polygons.

For the python implementation, please see the `test_integration_over_whole_sphere.py` and `test_one_eighth_area.py` for example.
Please import `compute_sphere_quadrature` as a module that provides two public interfaces.
One is `compute_sphere_quadrature` for the generation of quadrature points and corresponding weights on the sphere, which could be reused for integration of different functions.
The other is `spherical_integration` for integration over some spherical polygons.
We provide the quadrature rule table for the reference triangle on a 2D plane in `quadrature_rule.py`, which can be rewritten to set up your favorite quadrature rule.

## Copyright and Licenses ##

BSD 3-Clause License

Copyright (c) 2021, NumGeom Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

## How to Cite `ARPIST` ##

If you use `ARPIST` in your research for nonsingular systems, please cite the `ARPIST` paper:

```bibtex
@article{li2022arpist,
  title={ARPIST: Provably Accurate and Stable Numerical Integration over Spherical Triangles},
  author={Li, Yipeng and Jiao, Xiangmin},
  journal={arXiv preprint arXiv:2201.00261},
  year={2022}
}
```

## Contacts ##

- Yipeng Li, <yipeng.li@stonybrook.edu>, <jamesonli1313@gmail.com>
- Xiangmin Jiao, <xiangmin.jiao@stonybrook.edu>, <xmjiao@gmail.com>

