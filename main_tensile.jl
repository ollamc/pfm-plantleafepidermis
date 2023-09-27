using SharedArrays
using DelimitedFiles
using .FEM_IPOPT

### Initialization
ndof = 3
ninpt = 4
nstatevar = 10

timestep = 100
file1 = "coord.txt"
file2 = "connec.txt"

cpus = 12

# Material Props
E = 2000
ν = 0.35
Gc = 0.3
lc = 0.21
xk = 10^-10

#Interface Props
Gc_i = 2.7/6
lc_i = 0.17

# Additional parameters
energySplit = false          # Activate Miehe's Stress Decomposition
quasiMono = false            # Activate quasi-monolithic scheme (Heister, Wick) NOT PERTINENT IF USING IPOPT (FULLY MONOLITHIC)

# Output parameters
output = [1, 5]            # Output reaction force every output[1] timestep, output solution every output[2] timestep (VTK)

props = SharedArray([E, ν, Gc, lc, xk])
props_i = SharedArray([E, ν, Gc*0.5, lc, xk])

# Construct Dirichlet [Node, DOF, value]
nset1 = readdlm("nset1.txt", ',', Int64)
nset2 = readdlm("nset2.txt", ',', Int64)
nset3 = readdlm("nset3.txt", ',', Int64)
nset4 = readdlm("nset4.txt", ',', Int64)

dirichlet = SharedArray(zeros(2*length(nset1)+length(nset2)+length(nset3),3))

# Impose 0.0 to DDL 1 on nset1
dirichlet[1:length(nset1),1] = nset1
dirichlet[1:length(nset1),2] .= 1
dirichlet[1:length(nset1),3] .= 0.0

# Impose 0.0 to DDL 2 on nset1
dirichlet[length(nset1)+1:2*length(nset1),1] = nset1
dirichlet[length(nset1)+1:2*length(nset1),2] .= 2
dirichlet[length(nset1)+1:2*length(nset1),3] .= 0.0

# Impose 0.0 to DDL 2 on nset1
dirichlet[2*length(nset1)+1:2*length(nset1)+length(nset2),1] = nset2
dirichlet[2*length(nset1)+1:2*length(nset1)+length(nset2),2] .= 2
dirichlet[2*length(nset1)+1:2*length(nset1)+length(nset2),3] .= 1.5

# Impose 0.0 to DDL 3 on nset3
dirichlet[2*length(nset1)+length(nset2)+1:end,1] = nset3
dirichlet[2*length(nset1)+length(nset2)+1:end,2] .= 3
dirichlet[2*length(nset1)+length(nset2)+1:end,3] .= 0.0

# List nodes on which the Reaction Force will be extracted
outputRF = dirichlet[1:3,1]


interface_dirichlet  = zeros(length(nset4),1)
interface_dirichlet[1:length(nset4),1] = nset4
interface_dirichlet  = SharedArray(interface_dirichlet)

coord, connec = load(file1, file2)
coord = SharedArray(coord)
connec = SharedArray(connec)

# Initialize statevariable array
statevar = zeros(Float64,size(connec,1),ninpt,nstatevar)

# Assign number (adress) to each DOFs
numer = SharedArray(ordering(coord, dirichlet, ndof))

print("Regularizing Interface \n")
α = smear_interface(coord, connec, interface_dirichlet, props_i, cpus)
print("Regularizing Done! \n")

print("Solving Problem \n")
@time U = solveNEWTON(coord, connec, dirichlet, numer, α, props, props_i, timestep, statevar, cpus, energySplit, output, outputRF)
