using Gadfly
using DataFrames
using Optim

const λ = 1

# Import data
df = readtable("../data/ex2data2.txt", separator = ',', header = false,
               names = [:test1, :test2, :res])

# Add one label column
df[:result] = map(df[:res]) do x
    if x == 1
        return "accepted"
    else
        return "rejected"
    end
end

# Visualizing the data
p3 = plot(df, x = :test1, y = :test2, color = :result,  
          Scale.color_discrete_manual(colorant"deep sky blue",
                                      colorant"light pink"))

img = SVG("plot3.svg", 6inch, 4inch)
draw(img, p3)

# Construct feature matrix
function get_new_row(x1, x2)
    new_row = [1]
    for i in 1:6
        for j in 0:i
            new_row = hcat(new_row, [x1^(i-j) * x2^j])
        end
    end
    return new_row
end

# Since Julia cannot delete row from a matrix, we cant use one dummy row.
# To use `vcat` we need to build a real first row first.
feature = get_new_row(df[:test1][1], df[:test2][1])

# Concatenate new row to the feature matrix
for r in 2:size(df, 1)
    x1 = df[:test1][r]
    x2 = df[:test2][r]
    feature = vcat(feature, get_new_row(x1, x2))
end

# Construct value matrix
value = convert(Array, df[:res])

# Define the logistic function, which supports both one-example-vector, or a
# training-set-matrix. x would be row vector here, and this function assumes 
# the first entry of the row is 1. If x is a training-set-matrix, then each 
# row is one example, and this function returns a n*1 matrix. If x is 
# one-example-vector, this function would return a Float rather than an array.
function h(Θ, x)
    if size(x, 1) == 1
        return 1 / (1 + e ^ -(x * Θ)[1])
    else
        return result = map(z -> 1 / (1 + e ^ z), -(x * Θ))
    end
end

# Define the cost function. Regularization is used here. 
function cost(Θ)
    hypo = h(Θ, feature)
    m = size(hypo, 1)
    pre = ((-value' * log(hypo) - (1 - value)' * log(1 - hypo)) / m)[1]
    # We do not regularize the first θ
    return pre + λ / (2 * m) + sum(map(θ -> θ^2, Θ[2:end]))
end

# Define the gradient function for cost function. Regularization is used.
function g!(Θ, storage)
    m =  size(feature, 1)
    result = (feature' * (h(Θ, feature) - value)) / m
    storage[1] = result[1]
    result_reg = result + (λ / m) * Θ
    for i in 2:length(result_reg)
        storage[i] = result_reg[i]
    end
end

# Find parameters Θ minimizing the cost function
res = optimize(cost, g!, repeat([0.1], inner = size(feature, 2)))
mini_Θ = Optim.minimizer(res)
mini_cost = Optim.minimum(res)

println("The optimal Θ is $(mini_Θ)")
println("The minimal cost is $(mini_cost)")

# Function to predict the probability of being accepted
prob(test1, test2) = h(mini_Θ, get_new_row(test1, test2))

function get_func_expression(Θ)
    out = "$(Θ[1])"
    r = 2
    for i in 1:6
        for j in 0:i
            out = string(out, " + ", "($(Θ[r])) * x1^$(i-j) * x2^$j")
            r += 1
        end
    end
    return out
end
