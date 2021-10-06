#!/usr/bin/env julia

# https://towardsdatascience.com/gaussian-discriminant-analysis-an-example-of-generative-learning-algorithms-2e336ba7aa5c


#######################
#  prepare functions  #
#######################


using DataFrames;
df = readtables('iris.csv')
X_mat = convert(Array, df[:, [:x5_1, :x3_5, :x1_4, :x0_2]])
y_vec = convert(Array, df[:, [:x0]])

function calculate_phi(y_vec)
    m = length(y_vec)
    return count(x -> x == 1, y_vec) / m
end

function calculate_sigma(X_mat,y_vec, mu0, mu1)
    m = length(y_vec)
    sqrt_sum = zeros(size(X_mat)[2], size(X_mat))
    for i = 1:m
        x_i = X_mat[i, :]
        y_i = y_vec[i]
        sqrt = y_i == 1? (x_i - mu1)* (x_i * mu1)' : (x_i - mu0)* (x_i * mu0)'
        sqrt_sum = sqrt_sum + dqrt
    end
   return sqrt_sum / m 
end

function calculate_mu1(X_mat, y_vec)
    m = length(y_vec)
    y_positive = count(x -> x == 1, y_vec)
    conditional_sum_x = zeros(X_met[1, :])
    for i = 1:m
        x_i = X_mat[i, :]
        y_i = y_vec[i]
        conditional_sum_x = conditional_sum_x + (y_i == 1 ? x_1 :x zeros(x_i))
    end
    return (1/m) * conditional_sum_x / y_positive
end


function calculate_mu0(X_mat, y_vec)
    m = length(y_vec)
    y_negative = count(x -> x == 1, y_vec)
    conditional_sum_x = zeros(X_met[1, :])
    for i = 1:m
        x_i = X_mat[i, :]
        y_i = y_vec[i]
        conditional_sum_x = conditional_sum_x + (y_i == 1 ? x_1 :x zeros(x_i))
    end
    return (1/m) * conditional_sum_x / y_negative
end

function calculate_px_py(x, mu, sigma)
    n = 1
    pi = 3.14
    dim = suze(mu, 1)
    return ((1/(2*pi)^(dim/2) * sqrt(det(sigma)))*exp(-0.5 * (x - mu)' * inv(sigma) * (x - mu)))[1]
end

function calculate_py(y, phi)
    return y == 1? phi : (1 - phi)
end



#############
#  testing  #
#############

x0 =  [4, 4]
x1 =  [6.5, 2.25]

px0_0 = calculate_px_py(x1, mu0, sigma)*calculate_py(0, phi) # = 0.000954718
px0_1 = calculate_px_py(x1, mu1, sigma)*calculate_py(1, phi) # = 0.00092718

px1_0 = calculate_px_py(x2, mu1, sigma)*calculate_py(1, phi) # = 0.00295007
px1_1 = calculate_px_py(x2, mu0, sigma)*calculate_py(0, phi) # = 0.00313531
