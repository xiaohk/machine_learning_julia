include("display.jl")
# We already have variable images and numbers from that module

# Import weights data
data = matread("../data/ex4weights.mat")
pre_Θ1, pre_Θ2 = data["Theta1"], data["Theta2"]
λ = 1
ϵ = 0.12

# Add value 1 column in the images matrix imported in `display.jl`
feature = hcat([1 for i in 1:size(images, 1)], images)

#=  Create output layer for this NN model, for this problem we would have 10
    output units, since we are using the 1 of N encoding method. Each unit is
    one entry, so we need to re-encode the output matrix. We put each output
    as one 10 by 1 matrix, so the output matrix would be a 10 by 5000 matrix.
=#
function make_output_layer(digit::Float64)
    tem = zeros(10)
    tem[convert(Int64, digit)] = 1.0
    tem
end

output = zeros(10)
output[convert(Int64, numbers[1])] = 1.0
for n in numbers[2:end]
    output = hcat(output, make_output_layer(n))
end


# x can be either a number or an array
function sigmoid(z)
    if size(z) == ()
        return 1 / (1 + e ^ (-z))
    else
        tem = Array{Float64}(length(z))
        for i in 1:length(z)
            tem[i] = 1 / (1 + e ^ (-z[i]))
        end
    end
    return tem
end

# The gradient function of sigmoid function
sigmoid_prime(z) = sigmoid(z) * (1 - sigmoid(z))

# Input layer unit x can be either a row vector or a column vector
function h(Θ1, Θ2, x)
    # x is the input layer, we want to get the hidden layer first
    z2 = size(x,2) == 1 ? Θ1 * x : Θ1 * x'
    a2 = sigmoid(z2)

    # Get the output layer
    # Add bias unit to a2
    unshift!(a2, 1)
    z3 = Θ2 * a2
    a3 = sigmoid(z3)
    
    return a3
end

# Input layer unit x can be either a row vector or a column vector
function predict(Θ1, Θ2, x)
    convert(Float64, find(p -> p == maximum(h(Θ1, Θ2, x)), h(Θ1, Θ2, x))[1])
end

# Cost function for this NN model
function cost(Θ1, Θ2)
    # Computing the first term of the cost function
    first_term = 0.0
    m = size(feature, 1)
    for i in 1:m
        p = h(Θ1, Θ2, feature[i,:])
        first_term = first_term + (-1) * output[:,i]' * 
                     log(p) - (1 - output[:,i])' * log(1 - p)
    end
    first_term /= m
    
    # Computing the second term of the cost function (regularization term)
    # We don't regulate the bias weights, so we skip the first column of each
    # weight matrix
    second_term = λ / (2 * m) * (sum(Θ1[:, 2:end].^2) + sum(Θ2[:, 2:end].^2))

    return first_term + second_term
end

# Using random initialization to initialize the weights
# Return tuple (Θ1, Θ2)
function init_weights()
    Θ1 = Array{Float64,2}(25, 401)
    Θ2 = Array{Float64,2}(10, 26)

    # Init Θ1
    for i in 1:size(Θ1, 1)
        for j in 1:size(Θ1, 2)
            Θ1[i, j] = rand(collect(-1 * ϵ : 0.0001 : ϵ))
        end
    end
    
    # Init Θ2
    for i in 1:size(Θ2, 1)
        for j in 1:size(Θ2, 2)
            Θ2[i, j] = rand(collect(-1 * ϵ : 0.0001 : ϵ))
        end
    end

    return (Θ1, Θ2)
end




