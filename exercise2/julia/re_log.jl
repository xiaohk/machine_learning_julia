using Gadfly
using DataFrames
using Optim

λ = 1

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
    m = size(feature, 1)
    hypo = h(Θ, feature)
    pre = ((-value' * log(hypo) - (1 - value)' * log(1 - hypo)) / m)[1]
    # We do not regularize the first θ
    return pre + ((λ / (2 * m)) * sum(Θ[2:end] .^ 2))
end    

# Define the gradient function for cost function. Regularization is used.
function g!(Θ, storage)
    m =  size(feature, 1)
    hypo = h(Θ, feature)
    result = (feature' * (h(Θ, feature) - value)) / m
    # No regularization for Θ[1]
    storage[:] = result + (λ / m) * [0; Θ[2:end]]
end


# Find parameters Θ minimizing the cost function
res = optimize(cost, g!, repeat([0.5], inner = size(feature, 2)))
mini_Θ = Optim.minimizer(res)
mini_cost = Optim.minimum(res)

println("The optimal Θ is $(mini_Θ)\n")
println("The minimal cost is $(mini_cost)")

# Function to predict the probability of being accepted
prob(test1, test2) = h(mini_Θ, get_new_row(test1, test2))

# Plot the decision boundary
function decision(x1::Float64, x2::Float64, a::Array{Float64})
    dot(a, [1, x1^1*x2^0, x1^0*x2^1, x1^2*x2^0, x1^1*x2^1, x1^0*x2^2, 
            x1^3*x2^0, x1^2*x2^1, x1^1*x2^2, x1^0*x2^3, x1^4*x2^0, x1^3*x2^1,
            x1^2*x2^2, x1^1*x2^3, x1^0*x2^4, x1^5*x2^0, x1^4*x2^1, x1^3*x2^2,
            x1^2*x2^3, x1^1*x2^4, x1^0*x2^5, x1^6*x2^0, x1^5*x2^1, x1^4*x2^2, 
            x1^3*x2^3, x1^2*x2^4, x1^1*x2^5, x1^0*x2^6])
end

l1 = layer(df, x = :test1, y = :test2, color = :result, Geom.point)
l2 = layer(z = (x1,x2) -> decision(x1, x2, mini_Θ), 
           x = linspace(-1.0, 1.5, 100),
           y = linspace(-1.0, 1.5, 100),
           Geom.contour(levels = [0.0]),
           Theme(line_width = 1pt))

coord = Coord.cartesian(xmin=-1.0, xmax=1.5, ymin=-1.0, ymax=1.5)

p4 = plot(l1, l2, coord, Scale.color_discrete_manual(colorant"deep sky blue",
                                                     colorant"light pink"))

img = SVG("plot4.svg", 6inch, 4inch)
draw(img, p4)

# Plot multiple graphs using different λ
function cost_and_gradient(Θ, λ)
    m = size(feature, 1)
    hypo = h(Θ, feature)
    return (Θ::Array) -> begin
        pre = ((-value' * log(hypo) - (1 - value)' * log(1 - hypo)) / m)[1]
        return pre + ((λ / (2 * m)) * sum(Θ[2:end] .^ 2))
    end, (Θ::Array, storage::Array) -> begin
        result = (feature' * (hypo - value)) / m
        storage[:] = result + (λ / m) * [0; Θ[2:end]]
    end
end

cost2, g2! = cost_and_gradient(zeros(28), 0.5)




