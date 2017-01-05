using Gadfly
using Optim
using DataFrames

df = readtable("../data/ex2data1.txt", header = false, 
               names = [:exam1, :exam2, :admit], separator = ',')

# Add one more column for the sake of plot
df[:admission] = map(df[:admit])do x
    if x == 1
        return "Admitted"
    else
        return "Not Admitted"
    end
end


p1 = plot(df, x = :exam1, y = :exam2, color = :admission,
         Scale.color_discrete_manual(colorant"deep sky blue",
                                     colorant"light pink"))

img = SVG("plot1.svg", 6inch, 4inch)
draw(img, p1)


# Set up training sets
feature = ModelMatrix(ModelFrame(admit ~ exam1 + exam2, df)).m
value = convert(Array, df[:admit])

# Define the logistic function, which supports both one-example-vector, or 
# an example matrix. x would be row vector here, and this function assumes the
# first entry of the row is 1. If x is an example matrix, then each row is one 
# example, and this function returns a n*1 matrix. If x is one-example-vector,
# this function would return a Float.
function h(Θ, x)
    if size(x, 1) == 1
        return 1 / (1 + e ^ -(x * Θ)[1])
    else
        return result = map(z -> 1 / (1 + e ^ z), -(x * Θ))
    end
end

# Define the cost function. Parameter x should be a matrix, where each row 
# represents a training example. Parameter Y should be a column vector, 
function cost(Θ)
    hypo = h(Θ, feature)
    ((-value' * log(hypo) - (1 - value)' * log(1 - hypo)) / size(hypo, 1))[1]
end

# Define the gradient function for cost function
function g!(Θ, storage)
    result = (feature' * (h(Θ, feature) - value)) / size(feature, 1)
    for i in 1:length(result)
        storage[i] = result[i]
    end
end

# Use Optim.jl to minimize the cost function 
res = optimize(cost, g!, [0.1, 0.1, 0.1])
mini_Θ = Optim.minimizer(res)
mini_cost = Optim.minimum(res)

println("The optimal Θ is $(mini_Θ)")
println("The minimal cost is $(mini_cost)")

# Plot the points with the decision boundary
decision(x) = (mini_Θ[1] + mini_Θ[2] * x) / -mini_Θ[3]
l1 = layer(decision, 0, 100, Geom.line)
l2 = layer(df, x = :exam1, y = :exam2, color = :admission, Geom.point)
p2 = plot(l1, l2, Scale.color_discrete_manual(colorant"deep sky blue",
                                              colorant"light pink"),
          Coord.cartesian(xmin = 0, xmax = 100, ymin = 0, ymax = 100))
img = SVG("plot2.svg", 6inch, 4inch)
draw(img, p2)


# Function to predict the probability of admission
prob(exam1, exam2) = h(mini_Θ, [1 exam1 exam2])
