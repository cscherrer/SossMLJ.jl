function check_rows(a::AbstractArray, b::AbstractArray)
    if size(a, 1) != size(b, 1)
        throw(DimensionMismatch("the input arrays do not have the same size"))
    end
    return nothing
end
