module CommunityDetection
using Graphs
using ArnoldiMethod: LR, SR
using LinearAlgebra: I, Diagonal
using Clustering: kmeans

export community_detection_nback, community_detection_bethe, community_detection_greedy_modularity

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
    λ, eigv = Graphs.LinAlg.eigs(B, nev=k+1, which=LR())
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
    λ, eigv = Graphs.LinAlg.eigs(Hr, which=SR(), nev=min(kmax, nv(g)))

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


function community_detection_greedy_modularity(g::AbstractGraph)
    n = length(vertices(g))
    c = Vector(1:n)
    cs = Vector()
    qs = fill(-1., n)
    Q, e, a = compute_modularity(g, c)
    push!(cs, c)
    qs[1] = Q
    for i=1:n-1
        Q = modularity_greedy_step!(g, Q, e, a, c)
        push!(cs, c)
        qs[i+1] = Q
    end
    imax = argmax(qs)
    return rewrite_class_ids(cs[imax])
end

function modularity_greedy_step!(g::AbstractGraph, Q::Float64, e::Matrix{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat},  c::AbstractVector{<:Integer})
    m = 2 * length(edges(g))
    n = length(vertices(g))
    dq_max = -1
    tried = Set{Tuple{Int64, Int64}}()
    to_merge::Tuple{Integer, Integer} = (0,0)
    tried = Set()
    for edge in edges(g)
        u = min(src(edge), dst(edge))
        v = max(src(edge), dst(edge))
        if !((u, v) in tried)
            push!(tried, (u,v))
            dq = 2* (e[u,v] / m - a[u]*a[v] / m^2)
            if dq > dq_max
                dq_max = dq
                to_merge = (c[u], c[v])
            end
        end
    end
    c1, c2 = to_merge
    for i=1:n
        e[c1, i] += e[c2, i]
    end
    for i=1:n
        if i == c2
            continue
        end
        e[i, c1] += e[i, c2]
    end
    a[c1] = a[c1] + a[c2]
    for i=1:n
        if c[i] == c2
            c[i] = c1
        end
    end
    return Q
end


function compute_modularity(g::AbstractGraph, c::AbstractVector{<:Integer})
    Q = 0
    m = length(edges(g)) * 2
    n_groups = maximum(c)
    a = zeros(n_groups)
    e = zeros(n_groups, n_groups)
    for u in vertices(g)
        for v in neighbors(g, u)
            if c[u] == c[v]
                Q += 1
                e[c[i], c[j]] += 1
            end
            a[c[u]] += 1
        end
    end
    Q *= m
    for i=1:n_groups
        Q -= a[i]^2
    end
    Q /= m^2
    return Q, e, a
end

function rewrite_class_ids(v::AbstractVector{<:Integer})
    d = Dict()
    vn = zeros(Int64, length(v))
    for i=1:length(v)
        if !(v[i] in keys(d))
            d[v[i]] = length(d) + 1
        end
        vn[i] = d[v[i]]
    end
    return vn

end

end #module
