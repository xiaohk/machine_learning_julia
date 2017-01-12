include("./display.jl")
using Optim

#= 
Data is imported in display.jl. The variable name is `images`, `numbers` and
`Weights`.
=#

# There are 10 classifiers in total
const NUM = 10

# Transform the images images into a standard feature matrix
feature = hcat([1 for i in 1:size(images, 1)], images)

# Define the logistic function, which supports both one-example-vector, or a
# training-set-matrix. x can be either a row vector or a column vector if x is
# just one training example, and this function assumes the first entry of the 
# vector is 1. If x is a training-set-matrix, then each row is one example, 
# and this function returns a n*1 matrix. If x is one-example-vector, this 
# function would return a Float rather than an array.
function h(Θ, x)
    # Single row vector
    if size(x, 1) == 1
        return 1 / (1 + e ^ -(x * Θ)[1])
    # Single column vector
    elseif size(x, 2) == 1
        return 1 / (1 + e ^ -(x' * Θ)[1])
    # Training set matrix
    else
        return result = map(z -> 1 / (1 + e ^ z), -(x * Θ))
    end
end

# `value` is  different for each classifier. One way to deal with it is to 
# pass `value` in a single cost() or g!() as an argument. However, `g!()` does 
# not support this method, so for the sake of consistence, we use closure here.
# The code below is modified a little bit from the ex2, now it takes `value` as
# an argument,
function cost_and_gradient(Θ, value, λ)
    m = size(feature, 1)
    return (Θ::Array) -> begin
        hypo = h(Θ, feature)
        pre = ((-value' * log(hypo) - (1 - value)' * log(1 - hypo)) / m)[1]
        return pre + ((λ / (2 * m)) * sum(Θ[2:end] .^ 2))
    end, (Θ::Array, storage::Array) -> begin
        hypo = h(Θ, feature)
        result = (feature' * (hypo - value)) / m
        storage[:] = result + (λ / m) * [0; Θ[2:end]]
    end
end


# Construct the value vector for each classifier. value vector consists only
# 1 and 0. 1 represents it is in this classification while 0 means not.
get_value(k) = [i == k ? 1 : 0 for i in numbers]

# Start the training
# Preallocate the big matrix for all parameter Θ, where each row is a parameter
# for that classifier
Θ = Array{Float64}(NUM, size(feature, 2))

for k in 1:NUM
    # Get the local mini_θ
    cost, g! = cost_and_gradient(zeros(size(feature, 2)), get_value(k), 1)
    res = optimize(cost, g!, zeros(size(feature, 2)))
    mini_Θ = Optim.minimizer(res)
    
    # Build mini_θ into the big matrix Θ
    for col in 1:length(mini_Θ)
        Θ[k, col] = mini_Θ[col]
    end
end

# Multi-class prediction. Parameter x could be either a row vector (1 * 400) or
# a column vector (400 * 1).
function predict(x)
    # Compute the probabilities of each classes
    sigmoid(z) =  1 / (1 + e ^ (-z))
    xΘ = size(x, 1) == 1 ? Θ * x' : Θ * x
    proba = Array{Float64}(NUM)
    for i in 1:length(proba)
        proba[i] = sigmoid(xΘ[i])
    end
    # Return prediction
    return find(p -> p == maximum(proba), proba)[1]
end

# Test using result-known examples, return the correct rate
function predict_test(feature, value)
    correct_num = 0
    for t in 1:size(feature, 1)
        if predict(feature[t, :]) == value[t]
            correct_num += 1
        end
    end
    return correct_num / size(feature, 1)
end
