using MAT
using Images, Colors, ImageView

# Import the data from the .mat files
data = matread("../data/ex4data1.mat")
images, numbers = data["X"], data["y"]

# The width of the small image
const WIDTH = convert(Int64, sqrt(length(images[1, :])))

# Reshape one row of the training example into a square matrix
function restore(row::Array{Float64, 1})
    width = convert(Int64,sqrt(length(row)))
    return  clamp01nan.(reshape(row, width, width))
end

function move_image!(row::Int64, col::Int64, square::Array{Float64, 2}, 
                    im::Array{Float64, 2})
    for i in 0:WIDTH-1
        for j in 0:WIDTH-1
            square[row + i, col + j] = im[i + 1, j + 1]
        end
    end
end

get_image(im_array::Array) = colorview(Gray, im_array)

function min_perimeter(total::Int64)
    min_a = min_b = start = convert(Int64, ceil(sqrt(total)))
    peri = Inf
    for a in convert(Int64, floor(start/2)):start
        for b in convert(Int64, floor(start/2)):start
            if (a + b) * 2 < peri && total <= a * b
                min_a, min_b = a, b
                peri = (a + b) * 2
            end
        end
    end
    return min_a, min_b
end

# Display the images. `rows` is an array consisting the row number of the 
# images to display. The default is to randomly display 100 images.
function display_image(rows::Array = [])
    # Set up width and other variables
    if rows == []
        rows = rand(1:size(images, 1), 100)
    end

    if (tem_1 = sqrt(length(rows))) == (tem_2 = convert(Int64, ceil(tem_1)))
        width = len = tem_2
    else
        len, width =  min_perimeter(length(rows))
    end

    fill_num = length(rows) - width * len             

    # Preallocate a big square matrix
    square = Array{Float64}(len * WIDTH, width * WIDTH)
    row_cand = [i * WIDTH + 1 for i in 0:len-1]
    col_cand = [i * WIDTH + 1 for i in 0:width-1]
    i = 1
    
    # Moving small images into the big square in a loop
    for row in row_cand
        for col in col_cand
            if i <= length(rows)
                move_image!(row, col, square, restore(images[rows[i], :]))
                i += 1
            else
                # Fill the extra black squares
                move_image!(row, col, square, zeros(WIDTH, WIDTH))
            end
        end
    end

    imshow(get_image(square))
    return square
end


