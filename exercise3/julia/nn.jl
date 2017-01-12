include("display.jl")

# Import weights data
data = matread("../data/ex3weights.mat")
Θ1, Θ2 = data["Theta1"], data["Theta2"]

# Add value 1 column in the images matrix imported in `display.jl`
feature = hcat([1 for i in 1:size(images, 1)], images)

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

# Input layer unit x can be either a row vector or a column vector
function predict(x)
    # x is the input layer, we want to get the hidden layer first
    z2 = size(x,2) == 1 ? Θ1 * x : Θ1 * x'
    a2 = sigmoid(z2)

    # Get the output layer
    # Add bias unit to a2
    unshift!(a2, 1)
    z3 = Θ2 * a2
    a3 = sigmoid(z3)
    
    return find(p -> p == maximum(a3), a3)[1]
end

# Test using result-known examples, return the correct rate. It also supports
# returning the example which classifier fails to classify with the wrong 
# class prediction.
function predict_test(feature::Array{Float64, 2}, value::Array{Float64, 1})
    correct_num = 0
    failed = Dict{Int64, Int64}
    for t in 1:size(feature, 1)
        pred =  predict(feature[t, :])
        if  pred == value[t]
            correct_num += 1
        else
            failed[t] = pred
        end
    end
    return correct_num / size(feature, 1), failed
end

