# using Gadfly

# Import data
size, num_bed, price = Float64[], Float64[], Float64[]
matrix_1d = Float64[]
matrix_size = Int64[]
open("data/ex1data2.txt", "r") do fp
    lines = readlines(fp)
    push!(matrix_size, length(lines))
    push!(matrix_size, length(split(lines[1], ',')))
    for line in lines
        append!(matrix_1d, 
                [parse(Float64, entry) for entry in split(line, ',')])
    end
end

# Store data into a matrix, where each row is one training example
matrix = reshape(matrix_1d, matrix_size[2], matrix_size[1])'

# Normalize each feature
function normalize(feature)
    bar, st = mean(feature), std(feature)
    feature = map((x) -> (x - bar) / st, feature)
    feature
end

for column in 1:length(matrix[1,:])-1
    matrix[:,column] = normalize(matrix[:,column])
end

# Linear Regression
const α = 0.01
const ITERATION = 15000

# Add a header column for convenience, and remove the price column
training = hcat([1 for i in 1:length(matrix[:,1])], matrix[:,1:end-1])
value = matrix[:,end]
Θ = [1.0 for i in 1:length(training[1,:])]

function gradient_descent(Θ, train, value, α)
    for i in 1:length(Θ)
        gradient = 0
        train_num = length(train[:,1])
        for row in 1:train_num
            gradient += (train[row,:]' * Θ - value[row]) * train[row,i]
        end
        Θ[i] -= (α/train_num) * gradient[1]
    end
end

function cost_function(Θ, train, value)
    cost = 0.0
    for row in 1:length(train[:,1])
        cost += (train[row,:]' * Θ - value[row])[1]^2
    end
    cost
end

for i in 1:ITERATION
    gradient_descent(Θ, training, value, α)
    #println(cost_function(Θ, training, value))
end

output = "Estimation: y = ($(Θ[1]))"
for i in 1:length(Θ) - 1
    output = string(output, " + ($(Θ[i + 1]))x$(i)")
end

println(output)
println("Squared Error: $(cost_function(Θ, training, value))")

        

