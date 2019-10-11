# 1D tests

using MATLAB, JuliaFEM, Plots, Optim, LinearAlgebra, DelimitedFiles

LOAD_PATH
path = push!(LOAD_PATH, "C:/Users/.../julia")
path = push!(LOAD_PATH, "C:/Users/.../julia/MATLAB")
using Module


source =  [1/3 , 2/3]
obs    = 0.1:0.2:0.9
alphaReg    = 1e-7
tau         = 1e-4


Nnodes = 2 .^(2:6)
Nnodes = Nnodes + convert(Array{Int64,1}, ones(length(Nnodes)))

# level = 5
FineNodes   =  range(0, stop = 1, length = Nnodes[end])
FineElem        = 1:1:size(FineNodes,1)-1
Elem_center = FineNodes[1:end-1] + diff(FineNodes)/2
k           = Gauss(Elem_center)

u      = zeros(size(FineNodes,1),size(source,1))
C      = zeros(size(obs,1),size(FineNodes,1),size(source,1))
d      = zeros(size(obs,1),size(source,1))

for i in 1:length(source)
    global u[:,i],~,~,~        = FEMmodel1D(FineNodes,FineElem,k,source[i])
    global C[:,:,i], d[:,i]    = MapToObservations(obs,u[:,i],FineNodes)
end


using Plots

plot(FineNodes,u[:,1])
plot!(FineNodes,u[:,2])

#deriviteve of the stifness matrix
dAdk  = zeros(length(FineNodes),length(FineNodes),length(FineElem))

for i = 1:length(FineElem)
    k            = zeros(length(FineElem))
    k[i]         = 1
    dAdk[:,:,i]  = FEMmodel1Ddadk(FineNodes,FineElem,k,0.5) #source has no affect
end


# start the multilevel
MaxLevel = length(Nnodes)
counter = 0
cost = zeros(MaxLevel*6)
time = 0
X      = zeros(length(FineElem),MaxLevel*6)
delta  = zeros(MaxLevel*6)
grad = zeros(size(X))


for level = 1:MaxLevel

    CoarseNodes       =  range(0, stop = 1, length = Nnodes[level])
    Elem        = 1:1:size(CoarseNodes,1)-1
    Elem_center = CoarseNodes[1:end-1] + diff(CoarseNodes)/2

    global res
    if level == 1
        println("Newton Trust Region: \\\\\\\\\\\\")
        println("Level number ",level)
        initial_x = ones(Nnodes[level] - 1)*0.5
    elseif level == MaxLevel
        println("Final level: \\\\\\\\\\\\")
        CoarseMidNodes       =  range(0, stop = 1, length = Nnodes[level-1])
        initial_x            = CoarseToFine(Optim.minimizer(res),CoarseMidNodes,FineNodes)
    else
        println("Level number ",level)
        CoarseMidNodes       =  range(0, stop = 1, length = Nnodes[level-1])
        initial_x            = CoarseToFine(Optim.minimizer(res),CoarseMidNodes,CoarseNodes)
    end

    if level != MaxLevel
        dAdk_level  = dAdkML(dAdk,CoarseNodes,FineNodes)
    else
        dAdk_level = dAdk
    end

    L = ComputeL(Elem_center)
    #println("Objective is equal to:", Objective(initial_x,FineNodes,FineElem,source,L,alphaReg,C,d,CoarseNodes))

    res = Optim.optimize(k -> Objective(k,FineNodes,FineElem,source,L,alphaReg,C,d,CoarseNodes),
                    k -> Grad(k,FineElem,FineNodes,source,C,d,dAdk_level,L,alphaReg,CoarseNodes),
                    k -> Hessian(k,FineElem,FineNodes,source,C,d,dAdk_level,L,alphaReg,tau,CoarseNodes),
                    initial_x,  method=NewtonTrustRegion(; initial_delta = 1.0,
                    delta_hat = 100.0, eta = 0.1, rho_lower = 0.25,
                    rho_upper = 0.75) ;show_trace = false,store_trace = true,
                    extended_trace = true,
                    x_tol = 1e-6, iterations = 5, inplace = false)

    global counter
    global cost
    global time
    global X
    global delta
    global grad
    cost_val = Optim.f_trace(res)
    cost[counter + 1: counter+ length(cost_val)] = cost_val
    println("costval length = ",length(cost_val))
    for i = 1:length(Optim.trace(res))
        X[1:(2^(level+1)),counter + i]    = Optim.trace(res)[i].metadata["x"]
        delta[counter + i]  = Optim.trace(res)[i].metadata["delta"]
        #grad[:,counter + i] = Optim.trace(res)[i].metadata["g(x)"]
    end



    counter  = counter + length(cost_val)
    time = time + Optim.trace(res)[end].metadata["time"]
    println("time = " , time)

end
