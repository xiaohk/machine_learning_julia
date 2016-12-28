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
matrix = reshape(matrix_1d, matrix_size[1], matrix_size[2] )
println(matrix)

# Normalize each feature

# Linear Regression

