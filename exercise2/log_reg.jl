# using gadfly
using DataFrames

df = readtable("data/ex2data1.txt", header = false, 
               names = [:ex1, :ex2, :admit], separator = ',')

p = plot(df, x = :ex1, y = :ex2, color=:admit,
         Scale.color_discrete_manual(colorant"deep sky blue",
                                     colorant"light pink"),
         Guide.manual_color_key("Legend", ["Failure", "Success"],
                                ["deepskyblue", "lightpink"]))
println(df)
