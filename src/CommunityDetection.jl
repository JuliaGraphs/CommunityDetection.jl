module CommunityDetection
using LightGraphs
using ArnoldiMethod: LR, SR
using LinearAlgebra: I, Diagonal
using Clustering: kmeans
using SparseArrays
using SimpleWeightedGraphs
using Random: shuffle

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
function community_detection_louvain(g::AbstractGraph; tol::Real = 1e-6)
    graph = WGraph(copy(g))   # Create a weighted graph
    comms = collect(1:nv(g))  # Assume at first that each vertex is a community

    improvement = true
    levels = Vector{Int}[]    # Hold onto the group identity at each pass through `make_pass`
    while improvement
        improvement = louvain_make_pass!(graph, comms, tol)  # one pass of moving nodes and aggregating into communities
        improvement || break

        # create a new aggregated graph
        Anew = louvain_aggregate_communities!(graph, comms)
        push!(levels, comms)
        graph = WGraph(Anew)
        comms = collect(1:nv(graph))
    end

    # unpack and return communities containing the original nodes
    for i in length(levels):-1:2
        for node in eachindex(levels[i-1])
            levels[i-1][node] = levels[i][levels[i-1][node]]
        end
    end
    return levels[1]
end


function louvain_make_pass!(graph::AbstractGraph, comms::Vector{Int}, tol::Real)
    W = sum(weights(graph)) # total weight of graph
    node_W = reshape(sum(weights(graph),dims=1), :)
    comm_W = zeros(maximum(comms))
    for i in 1:nv(graph)
        comm_W[comms[i]] += node_W[i]
    end

    # `node_comm_W[i]` is the total weight between node and community `i`
    node_comm_W = zeros(maximum(comms))

    made_progress = true
    improvement = false
    nb_moves = 0  # number of moves
    while made_progress
        best_gain = 0.0
        for i in 1:nv(graph)
            current_comm = comms[i]
            improvement = false
            best_gain = 0.0
            best_comm = current_comm

            # compute weights between node `i` and its neighboring communities
            for neighbor in neighbors(graph, i)
                neighbor_comm = comms[neighbor]
                if neighbor != i
                    node_comm_W[neighbor_comm] += 2weights(graph)[i, neighbor]
                else
                    node_comm_W[neighbor_comm] += weights(graph)[i, neighbor]
                end
            end

            # change of modularity if we remove node `i` from old community
            ΔM1 = (comm_W[current_comm] - 1.0) * node_W[i] / W - node_comm_W[current_comm]
            node_comm_W[current_comm] = 0

            # using `shuffle` to select one from multiple proposed communities which have the same maximum modularity gain
            for neighbor in shuffle(neighbors(graph,i))
                neighbor_comm = comms[neighbor]
                if node_comm_W[neighbor_comm] > 0 && neighbor_comm != current_comm
                    # change of modularity if we add node to new community
                    ΔM2 = node_comm_W[neighbor_comm] - comm_W[neighbor_comm] * node_W[i] / W
                    modularity_gain = ΔM1 + ΔM2
                    if modularity_gain > best_gain
                        best_gain = modularity_gain
                        best_comm = neighbor_comm
                    end
                    # this neighboring community has seen, so set the weight to zero to skip it next time
                    node_comm_W[neighbor_comm] = 0
                end
            end
            if best_comm != current_comm
                nb_moves += 1
                comms[i] = best_comm  # move node to new community
                comm_W[current_comm] -= node_W[i]  # remove node weight from old community
                comm_W[best_comm] += node_W[i]  # add node weight to new community
            end
        end

        # check whether the algorithm is making progress for this pass
        improvement = nb_moves > 0
        made_progress = improvement && best_gain/2.0/W > tol
    end
    return improvement
end


#Aggregate matrix communities.
function louvain_aggregate_communities!(graph::AbstractGraph,comms::Vector{Int})
    #Renumber communities starting from 1
    scomms = sort(unique(comms))
    comm_map = Dict(zip(scomms,collect(1:length(scomms))))
    for i = 1:length(comms)
        comms[i] = comm_map[comms[i]]
    end

    #Create a group identify matrix and aggregate into larger communities
    n = nv(graph)
    A = graph.weights
    G = sparse(1:n, comms, ones(Int,n))
    Anew = G'*A*G  #New weighted adjacency matrix
    return Anew
end

end #module
