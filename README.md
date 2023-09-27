# pfm-plantleafepidermis

**Finite element implementation of a variational phase-field model for fracture in plant leaf epidermis.**
<br />

The code is associated to the manuscript

> Amir J. Bidhendi, Olivier Lampron, Frédérick P. Gosselin, Anja Geitmann, "Microscale geometrical features in the plant leaf epidermis confer enhanced resistance to mechanical failure," Submitted to Nature Communications, doi: 10.1101/2022.12.10.519895.

A preprint of the manuscript is available here: https://www.biorxiv.org/content/10.1101/2022.12.10.519895v1

Most of the code was reused and adapted from https://github.com/ollamc/juliaPF associated to the publication

> O. Lampron, D. Therriault, and M. Lévesque, “An efficient and robust monolithic approach to phase-field quasi-static brittle fracture using a modified Newton method,” Computer Methods in Applied Mechanics and Engineering, vol. 386, p. 114091, Dec. 2021, doi: 10.1016/j.cma.2021.114091.

See the latter repository for more information on the dependancies and the usage.

The script `FEM_INTERFACE.jl` contains the implementation of the quasi-static phase-field fracture model while `main_tensile.jl` contains the pre-processing and settings of the problem. The latter calls the functions of `FEM_INTERFACE.jl`.

The _Geometries_ folder contains the finite element meshes and the files containing the boundary conditions used the different cell configurations studied in the manuscript.

This work was developped for the two aforementionned publications. If you use this work, please cite these papers.


