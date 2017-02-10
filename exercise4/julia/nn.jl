include("display.jl")
# We already have variable images and numbers from that module
using Optim

# Import weights data
data = matread("../data/ex4weights.mat")
pre_Θ1, pre_Θ2 = data["Theta1"], data["Theta2"]
λ = 1
ϵ = 0.12
HU_NUM = 25

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
sigmoid_prime(z) = sigmoid(z) .* (1 - sigmoid(z))

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

    return (first_term + second_term)[1]
end

# Using random initialization to initialize the weights
# Return tuple (Θ1, Θ2)
function init_weights()
    Θ1 = Array{Float64,2}(HU_NUM, size(feature, 2))
    Θ2 = Array{Float64,2}(size(output, 1), HU_NUM + 1)

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

# Rolling two matrix into one vector
rolling(Θ1, Θ2) = [Θ1[:]; Θ2[:]]

# Unrolling one vector into two matrices by given dimension
# The first matrix is i1 by j1 and the second one is i2 by j2
function unrolling(roll, i1, j1, i2, j2)
   return (reshape(roll[1:i1 * j1], i1, j1), 
           reshape(roll[i1 * j1 + 1:end], i2, j2))
end

#=  Backward propagation: we first use forward propagation to compute the 
    hidden units, then use backward propagation to update the weights based on
    the known error for each weights.
=#
function backward(Θ1, Θ2)
    # Initialize the accumulator
    Δ1 = zeros(size(Θ1))
    Δ2 = zeros(size(Θ2))
    
    # Iterate through the feature space
    m = size(feature, 1)
    for (i in 1:m)
        # Step 1: Forward computing
        a1 = feature[i, :]

        # x is the input layer, we want to get the hidden layer first
        z2 = Θ1 * a1
        a2 = sigmoid(z2)

        # Get the output layer
        # Add bias unit to a2
        unshift!(a2, 1)
        z3 = Θ2 * a2
        a3 = sigmoid(z3)

        # Step 2: Get the output error
        δ3 = a3 - output[:, i]

        # Step 3: Get the hidden weights error
        # I didn't use the gradient function here because I want the bias unit
        # to be added. Then it is more convenient to remove it in the next step.
        δ2 = Θ2' * δ3 .* (a2 .* (1 - a2))

        # Step 4: Accumulate those δ into Δ
        Δ2 = Δ2 + δ3 * a2'
        Δ1 = Δ1 + δ2[2:end] * a1'

    end

    # Then we add regularization
    D2 = hcat((Δ2[:, 1] ./ m), (Δ2[:, 2:end]./ m + (λ / m) .* Θ2[:, 2:end]))
    D1 = hcat((Δ1[:, 1] ./ m), (Δ1[:, 2:end]./ m + (λ / m) .* Θ1[:, 2:end]))

    return (D1, D2)
end

# We need to modify the gradient function a little bit to use Optim.jl
function g!(x::Vector, storage::Vector)
    # First unroll the parameters
    (Θ1, Θ2) = unrolling(x, HU_NUM, size(feature, 2), 
                           size(output, 1), HU_NUM + 1)

    # Get the gradient for those parameters
    (D1, D2) = backward(Θ1, Θ2)

    # Rolling the gradients and add to storage
    storage[:] = rolling(D1, D2)
end

# Similarly, we want to modify the original cost function to use Optim.jl
function cost_optim(x::Vector)
    # First unroll the parameters
    (Θ1, Θ2) = unrolling(x, HU_NUM, size(feature, 2), 
                           size(output, 1), HU_NUM + 1)

    # Call the cost function
    return cost(Θ1, Θ2)
end

# Check gradient using the slope method
function check_gradient(para::Array{Float64,1})
    ϵ = 1e-4
    temp = zeros(size(para)...)
    esti = zeros(size(para)...)
    
    # Iterate through the weights
    for i in size(para, 1)
        temp[i] = ϵ
        diff = (cost())



res = optimize(cost_optim, g!, rolling(init_weights()...), 
               Optim.Options(iterations = 2))
mini_Θ = Optim.minimizer(res)




