module Module2

using MATLAB, JuliaFEM, Plots, Optim, LinearAlgebra

export Objective
export Grad
export Hessian
export FEMmodel1D
export FEMmodel1Ddadk
export MapToObservations
export Gauss
export ComputeL
export NewtonMethod
export dAdkML
export CoarseToFine


function Objective(k,FineNodes,FineElem,source,L,alphaReg,C,d,CoarseNodes)
    LS = 0.0
    cost = zeros(1)
    k_Fine  = CoarseToFine(k,CoarseNodes,FineNodes)
    for i = 1:size(source,2)
        u,~,~,~ = FEMmodel1D(FineNodes,FineElem,k_Fine,source[i])
        ls = norm(C[:,:,i]*u - d[:,i])^2 # denotes leftside
        LS += ls
    end

    reg = alphaReg/2*transpose(k)*L*k
    cost = 1/2 * LS + reg
    #println("LS = " , LS," reg =  ", reg)
    return cost
end

function Grad(k,FineElem,FineNodes,source,C,d,dAdk,L,alphaReg,CoarseNodes)

    grad    = zeros(size(L,1))
    gls     = zeros(size(L,1))
    k_Fine  = CoarseToFine(k,CoarseNodes,FineNodes)

    for i in 1:length(source)
        u,A,~,f = FEMmodel1D(FineNodes,FineElem,k_Fine,source[i])
        # LU = lu(A)
        F     = C[:,:,i]/A*f #C[:,:,i]*(LU.P*(LU.U\(LU.L\f))) # C[:,:,ii]/A*f
        r     = F - d[:,i]
        z     = A'\(-C[:,:,i]'*r) #LU.P'*(LU.L'\(LU.U'\(-C[:,:,i]'*r))) # A'\(-C[:,:,ii]'*r)
        gls1   = zeros(size(L,1))

        for j in 1:length(gls1)
            gls1[j]  = (dAdk[:,:,j]*u)'*z
        end
        gls += gls1
    end

    greg = alphaReg * L * k
    grad = gls + greg
    return grad
end

function Hessian(k,FineElem,FineNodes,source,C,d,dAdk_level,L,alphaReg,tau,CoarseNodes)

    Hls     = zeros(length(k),length(k))
    gls     = zeros(length(k))
    gls1    = zeros(length(k))
    gls     = Grad(k,FineElem,FineNodes,source,C,d,dAdk_level,L,alphaReg,CoarseNodes)

    for i in 1:length(gls1)
        ei      = zeros(length(gls)); #creating the standard i-th unit vector
        ei[i]   = 1;
        gls2    = zeros(length(gls))
        gls2    = Grad(k + tau*ei,FineElem,FineNodes,source,C,d,dAdk_level,L,alphaReg,CoarseNodes)
        gls1    = gls - alphaReg*L*(k) # remove the regularization term
        gls2    = gls2 - alphaReg*L*(k + tau*ei) # remove the regularization term
        Hei     = (gls2 - gls1)/tau; # Hessian matrix - vector multiplication approximation
        for j in 1:length(gls1)
            ej       = zeros(length(gls1)); #creating the standard i-th unit vector
            ej[j]    = 1
            Hls[i,j]  = dot(Hei,ej)
        end
    end

    Hreg = alphaReg * L
    Hess = Hls + Hreg
    return Hess
end

function FEMmodel1D(Nodes,Elem,k,source)
    # computes 1 D FEM problem with Dirichlet BC)
    problem = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(stdout, Base.CoreLogging.Warn)) do
                    problem = Problem(Heat, "example Heat", 1)
                    end

    # add triangle elements to problem
    D = Dict(j => Nodes[j] for j in 1:length(Nodes)) #Elem[end])
    source_loc = findall(x->x <= source[1], Nodes)[end]

    for i = 1:Elem[end] #-1

            el = Element(Seg2,[Elem[i],Elem[i]+1])
            update!(el, "geometry", D)
            update!(el, "thermal conductivity", k[i])
            add_element!(problem, el)
            #if isempty(findall(x->x == i, source_loc[:]))==false
            #        update!(el, "heat source", 1)
            #end
    end

    time = 0.0
    assemble!(problem, time)

    K = Matrix(problem.assembly.K) # before adjusting BC
    K_Neumann = copy(K)

    K[1,:]     =  zeros(size(Nodes,1),1)
    K[end,:]   =  zeros(size(Nodes,1),1)
    K[1,1]     = 1
    K[end,end] = 1
    f = zeros(size(Nodes,1))
    f[convert(Int64,source_loc[1])] = 1

    u = K\f

    plot!(Nodes,u)
    return u,K,K_Neumann,f #K is Stifness with Dirichlet BC
end

function FEMmodel1Ddadk(Nodes,Elem,k,source)
    # computes 1 D FEM problem with Dirichlet BC)
    problem = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(stdout, Base.CoreLogging.Warn)) do
                    problem = Problem(Heat, "example Heat", 1)
                    end

    # add triangle elements to problem
    D = Dict(j => Nodes[j] for j in 1:length(Nodes)) #Elem[end])
    source_loc = findall(x->x <= source[1], Nodes)[end]

    for i = 1:Elem[end] #-1

            el = Element(Seg2,[Elem[i],Elem[i]+1])
            update!(el, "geometry", D)
            update!(el, "thermal conductivity", k[i])
            add_element!(problem, el)
            #if isempty(findall(x->x == i, source_loc[:]))==false
            #        update!(el, "heat source", 1)
            #end
    end

    time = 0.0
    assemble!(problem, time)

    K = Matrix(problem.assembly.K) # before adjusting BC

    K[1,:]     = zeros(size(Nodes,1),1) #always constant due to BC
    K[end,:]   = zeros(size(Nodes,1),1) #always constant due to BC
    K[:,1]     = zeros(size(Nodes,1),1)
    K[:,end]   = zeros(size(Nodes,1),1)
    return K #K is Stifness with Dirichlet BC
end

function MapToObservations(obs,u,Nodes)
    obs_loc = zeros(Int32,1,length(obs))
    [obs_loc[i] = findall(x->x <= obs[i], Nodes)[end] for i  in 1:length(obs)]
    C = zeros(Int32,length(obs),length(u))
    d = zeros(1,length(obs))
    for i = 1:length(obs)
        C[i,obs_loc[i]] = 1
        d[i]            = u[obs_loc[i]]
    end
    return C,d
end

function Gauss(Nodes)
    x = Nodes
    c = ones(size(Nodes,1))*0.45 #/6
    sig = (Nodes[1] + Nodes[end]) /2/6
    a = -((x-c).^2)/(2*sig^2)
    k = -exp.(a)*0.2 + ones(size(a,1))
    return k
end

function ComputeL(Elem_center)
    Elem        = 1:1:length(Elem_center)-1
    k           = ones(length(Elem))
    ~,~,L,~         = FEMmodel1D(Elem_center,Elem,k,1) #Stifness matrix with Neumann BC
    L = L/(1/(Elem_center[2]-Elem_center[1]))
    # source[i]=0.5 will not change the stifness matrix
    return L
end

function dAdkML(dAdk,Nodes,FineNodes)
    Nelem = length(Nodes)-1
    dadk_new = zeros(length(FineNodes),length(FineNodes),Nelem)
    Fine_Elem_center = FineNodes[1:end-1] + diff(FineNodes)/2
    counter = 1
    for i = 1:length(Fine_Elem_center)
        if Fine_Elem_center[i] < Nodes[counter + 1]
            dadk_new[:,:,counter] += dAdk[:,:,i]
        else
            counter = counter+1
            dadk_new[:,:,counter] += dAdk[:,:,i]
        end
    end
    return dadk_new
end

function CoarseToFine(k_Coarse,CoarseNodes,Nodes)

    if CoarseNodes == Nodes
        #println("True - Coarse=Fine")
        return k_Coarse
    end
    k = zeros(length(Nodes)-1)
    Fine_Elem_center = Nodes[1:end-1] + diff(Nodes)/2
    counter = 1
    for i = 1:length(Fine_Elem_center)
        if Fine_Elem_center[i] >= CoarseNodes[counter + 1]
            counter = counter + 1
        end
        k[i] = k_Coarse[counter]
    end
    return k
end

end
