module CommunityDetection
using LightGraphs
using ArnoldiMethod: LR, SR
using LinearAlgebra: I, Diagonal
using Clustering: kmeans
using SparseArrays
using SimpleWeightedGraphs

export community_detection_nback, community_detection_bethe, community_detection_louvain

"""
    community_detection_nback(g::AbstractGraph, k::Int)

Return an array, indexed by vertex, containing commmunity assignments for
graph `g` detecting `k` communities.
Community detection is performed using the spectral properties of the 
non-backtracking matrix of `g`.

### References
- [Krzakala et al.](http://www.pnas.org/content/110/52/20935.short)
"""
function community_detection_nback(g::AbstractGraph, k::Int)
    #TODO insert check on connected_components
    ϕ = real(nonbacktrack_embedding(g, k))
    if k == 1
        c = fill(1, nv(g))
    elseif k==2
        c = community_detection_threshold(g, ϕ[1,:])
    else
        c = kmeans(ϕ, k).assignments
    end
    return c
end

function community_detection_threshold(g::AbstractGraph, coords::AbstractArray)
    # TODO use a more intelligent method to set the threshold
    # 0 based thresholds are highly sensitive to errors.
    c = ones(Int, nv(g))
    # idx = sortperm(λ, lt=(x,y)-> abs(x) > abs(y))[2:k] #the second eigenvector is the relevant one
    for i=1:nv(g)
        c[i] = coords[i] > 0 ?  1 : 2
    end
    return c
end


"""
	nonbacktrack_embedding(g::AbstractGraph, k::Int)

Perform spectral embedding of the non-backtracking matrix of `g`. Return
a matrix ϕ where ϕ[:,i] are the coordinates for vertex i.

### Implementation Notes
Does not explicitly construct the `non_backtracking_matrix`.
See `Nonbacktracking` for details.

### References
- [Krzakala et al.](http://www.pnas.org/content/110/52/20935.short).
"""
function nonbacktrack_embedding(g::AbstractGraph, k::Int)
    B = Nonbacktracking(g)
    λ, eigv = LightGraphs.LinAlg.eigs(B, nev=k+1, which=LR())
    ϕ = zeros(ComplexF32, nv(g), k-1)
    # TODO decide what to do with the stationary distribution ϕ[:,1]
    # this code just throws it away in favor of eigv[:,2:k+1].
    # we might also use the degree distribution to scale these vectors as is
    # common with the laplacian/adjacency methods.
    for n=1:k-1
        v= eigv[:,n+1]
        ϕ[:,n] = contract(B, v)
    end
    return ϕ'
end



"""
    community_detection_bethe(g::AbstractGraph, k=-1; kmax=15)

Perform detection for `k` communities using the spectral properties of the 
Bethe Hessian matrix associated to `g`.
If `k` is omitted or less than `1`, the optimal number of communities
will be automatically selected. In this case the maximum number of
detectable communities is given by `kmax`.
Return a vector containing the vertex assignments.

### References
- [Saade et al.](http://papers.nips.cc/paper/5520-spectral-clustering-of-graphs-with-the-bethe-hessian)
"""
function community_detection_bethe(g::AbstractGraph, k::Int=-1; kmax::Int=15)
    A = adjacency_matrix(g)
    D = Diagonal(degree(g))
    r = (sum(degree(g)) / nv(g))^0.5

    Hr = Matrix((r^2-1)*I, nv(g), nv(g)) - r*A + D;
    #Hmr = Matrix((r^2-1)*I, nv(g), nv(g)) + r*A + D;
    k >= 1 && (kmax = k)
    λ, eigv = LightGraphs.LinAlg.eigs(Hr, which=SR(), nev=min(kmax, nv(g)))

    # TODO eps() is chosen quite arbitrarily here, because some of eigenvalues
    # don't convert exactly to zero as they should. Some analysis could show
    # what threshold should be used instead
    q = something(findlast(x -> (x < -eps()), λ), 0)
    k > q && @warn("Using eigenvectors with positive eigenvalues,
                    some communities could be meaningless. Try to reduce `k`.")
    k < 1 && (k = q)
    k <= 1 && return fill(1, nv(g))
    labels = kmeans(collect(transpose(eigv[:,2:k])), k).assignments
    return labels
end

"""
    community_detection_louvain(g::AbstractGraph; tol = 1e-6)

Perform fast-unfolding to maximize the modularity of a graph. `tol` is the minimum amount of improvement that should be made to consider a pass
to make progress.  Return a vector containing the vertex assignments.

### References
- [Blondel et al.](https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta)
"""
function community_detection_louvain(g::AbstractGraph;tol::Float64 = 1e-6)
    graph = WGraph(copy(g))   # Create a weighted graph
    comms = collect(1:nv(g))  # Assume at first that each vertex is a community

    improvement = true
    levels = []               # Hold onto the group identity matrices at each pass through `make_pass`
    while improvement


        improvement,new_comms = louvain_make_pass(graph,comms,tol)  #one pass of moving nodes and aggregating into communities

        if improvement == false
            break
        end

        comms = new_comms

        #create a new aggregated graph
        Anew, G = louvain_aggregate_communities(graph,comms)
        push!(levels,G)
        graph = WGraph(Anew)
        comms = collect(1:nv(graph))

    end

    # Now unpack and return communities containing the original nodes
    # i.e. G = G[1]*G[2]*...*G[end]  where G is a sparse matrix with dimensions n_nodes x n_communities
    rhs = levels[end]
    for j = length(levels) - 1:-1:1
        rhs = levels[j]*rhs
    end

    #Return communities in a vector
    labels = Array{Int64}(undef,rhs.m)  #intialize community vector
    for i = 1:rhs.m
        @assert sum(rhs[i,:]) == 1  #each vertex should stricly be in one community
        ind = rhs[i,:].nzind[1]     #get the index of the vertex
        labels[i] = ind
    end

    return labels
end

function louvain_make_pass(graph::AbstractGraph,communities::Vector{Int},tol::Float64)
    comm_list = copy(communities)
    current_modularity = modularity(graph,comm_list)
    made_progress = true
    nb_moves = 0                        #number of moves
    improvement = false
    while made_progress
        for i = 1:nv(graph)
            current_community = comm_list[i]
            improvement = false
            best_gain = 0.0

            #propose moving node i to a new community adjacent to this node
            possible_communities = unique([comm_list[neighbor] for neighbor in neighbors(graph,i)])  #NOTE Could probably improve performance here.  Could maintain a mapping of neighbors to avoid calling unique every iteration.
            best_comm = current_community   #assume best community is the current one at first

            #Find the best community to put vertex i into
            for p_comm in possible_communities
                proposed_communities = copy(comm_list)
                proposed_communities[i] = p_comm #propose to move i to a new community
                #NOTE Can instead calculate the modularity gain directly which should lead to speed ups.
                new_modularity = modularity(graph,proposed_communities)
                gain = new_modularity - current_modularity
                if gain > best_gain
                    best_comm = p_comm
                    best_gain = gain
                end
            end

            #Insert node into best community
            comm_list[i] = best_comm
            if best_comm != current_community
                nb_moves += 1
            end
        end

        new_modularity = modularity(graph,comm_list)

        #check whether the algorithm is making progress for this pass
        made_progress = nb_moves > 0 && (new_modularity - current_modularity) > tol
        current_modularity = new_modularity

        if nb_moves > 0
            improvement = true
        end
    end
    return improvement, comm_list
end

#Aggregate matrix communities.
function louvain_aggregate_communities(graph::AbstractGraph,comms::Vector{Int})
    #Renumber communities starting from 1
    scomms = sort(unique(comms))
    comm_map = Dict(zip(scomms,collect(1:length(scomms))))
    for i = 1:length(comms)
        comms[i] = comm_map[comms[i]]
    end

    #Create a group identify matrix and aggregate into larger communities
    n_comms = maximum(comms)
    n_nodes = nv(graph)
    A = graph.weights

    G = zeros(n_nodes,n_comms)
    G = SparseMatrixCSC(G)
    for i = 1:n_nodes
        G[i,comms[i]] = 1
    end

    Anew = G'*A*G  #New weighted adjacency matrix
    return Anew, G
end


end #module
