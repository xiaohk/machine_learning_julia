using Gadfly

# Import data
population = Float64[]
profit = Float64[]
open("data/ex1data1.txt", "r") do fp
    lines = readlines(fp)
    for line in lines
        temp = split(line, ',')
        push!(population, parse(Float64, temp[1]))
        push!(profit, parse(Float64, temp[2]))
    end
    println(population)
end

# Scatter plot
p = plot(x=population, y=profit, Guide.xlabel("Population"), 
         Guide.ylabel("Profit"))
img = SVG("initil.svg", 6inch, 4inch)
draw(img, p)

# Linear regression
const α = 0.01
const ITERATION = 15000
parameter = [1.0, 1.0]

#= Function to compute the squared error (cost) =#
function compute_cost(data_x, data_y, b1, b0)
    cost = 0
    for i = 1:length(data_x)
        cost += (b1 * data_x[i] + b0 - data_y[i]) ^ 2
    end
    cost / (2 * length(data_x))
end

#= Update b1 and b0 using the batch gradient descent algorithm =#
function gradient_descent!(data_x, data_y, parameter, α)
    gradient1 = gradient0 = 0 
    m = length(data_x)
    b1, b0 = parameter[1], parameter[2]

    for i = 1:m
        gradient1 += (b1 * data_x[i] + b0 - data_y[i]) * data_x[i]
        gradient0 += (b1 * data_x[i] + b0 - data_y[i])
    end
    (parameter[1], parameter[2]) = (b1 - (α / m * gradient1), 
                                    b0 - (α / m * gradient0))
end

for i = 1:ITERATION
    gradient_descent!(population, profit, parameter, α)
    println(parameter[1])
end

estimation(x) = parameter[1] * x + parameter[2]

println("Squared Error: $(compute_cost(population, profit, 
                                        parameter[1], parameter[2]))")
println("Estimation: y = $(parameter[1])x + ($(parameter[2]))")

# Plot the estimation line with real data
domain = [minimum(population), maximum(population)]
value = [estimation(x) for x in domain]

p = plot(layer(x=population, y=profit, Geom.point), 
         layer(x=domain, y =value, Geom.line), 
         Guide.xlabel("Poplulation"),
         Guide.ylabel("Profit"))
img = SVG("linear.svg", 6inch, 4inch)
draw(img, p)
