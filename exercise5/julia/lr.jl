using Gadfly
using Optim
using MAT

mat_data = matread("./data/ex5data1.mat")
X, Xval, yval, Xtest, ytest, y = mat_data["X"], mat_data["Xval"],
                                 mat_data["yval"], mat_data["Xtest"],
                                 mat_data["ytest"], mat_data["y"]

plot(x=X, y=y, Geom.point, Guide.title("Training Example"))


"""
Cost function for regression models (least mean squared)
"""
function cost{T1, T2, T3<:Real}(x::Array{T1}, y::Array{T2}, 
                                theta::Array{T3}, lambda=1)
    @assert size(x, 2) + 1 == size(theta, 1)
    
    m = size(x, 1)
    
    # Add the interception term to eamples
    x_full = hcat(ones(m), x)
    
    # Compute MSE
    y_hat = x_full * theta
    mse = sum((y_hat - y) .^ 2) / (2 * m)
    
    # Lasso Penalty
    penalty = (lambda / (2 * m)) * sum((theta .^ 2)[2:end])
    
    return mse + penalty
end


"""
Gradient function for the cost function above.
"""
function gradient{T1, T2, T3<:Real}(x::Array{T1}, y::Array{T2}, 
                                    theta::Array{T3}, lambda=1)
    @assert size(x, 2) + 1 == size(theta, 1)
    
    m = size(x, 1)
    
    # Add the interception term to eamples
    x_full = hcat(ones(m), x)
    
    y_hat = x_full * theta
    
    # First column of full x are ones
    grad_0 = sum(y_hat - y) / m
    
    grad_other = []
    for c in 1:size(x,2)
        append!(grad_other, sum((y_hat - y) .* x[:,c]) / m + (lambda / m) * theta[c + 1])
    end
    
    return vcat(grad_0, grad_other)
end


"""
Gradient function wrapper for Optim.jl.
"""
function grad!(theta, store; feature=X, y=y, lambda=1)
    store[:] = gradient(feature, y, theta, lambda)
end


"""
Train a slr model using given x and y, return training and validation MSE
"""
function get_cv_errors(x, y; lambda=1, normal=false)
    nx = normal ? normalize(x) : x

    # Train the model
    res = optimize(t -> cost(nx, y, t, 0), (t, s) -> 
                   grad!(t, s, feature=nx, y=y, lambda=0), 
                   zeros(size(nx, 2) + 1))
    best_theta = Optim.minimizer(res)
    
    # Compute the error
    train_error = cost(nx, y, best_theta, 0)
    
    # Ensure val matrix 
    val_poly = get_poly(Xval, size(x, 2))
    val_error = cost(normal ? normalize(val_poly) : val_poly, 
                     yval, best_theta, 0)
    
    return (train_error, val_error)
end


"""
Map a m*1 training matrix into m*p matrix, where each row is the polynomial term
of coresponding row.

For example, [3; 4] would be mapped to [3, 9, 27; 4, 16, 64] when p = 3.
"""
function get_poly{T<:Real}(x::Array{T}, p::Int64)
    result = x[:]
    for i in 2:p
        result = hcat(result, result[:,1] .^ i)
    end
    return result
end     


"""
Use mean, sd to normalize the feature matrix.
"""
normalize{T<:Real}(x::Array{T}) = (x .- mean(x)) ./ std(x)
