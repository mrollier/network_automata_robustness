## cell 1

using LinearAlgebra

# function to create tridiagonal circulant matrix
function tridiagonal_circulant(N, main, lower, upper)
    C = zeros(Float64, N, N)  # Create an N x N matrix initialized with zeros

    # Set the main diagonal
    for i in 1:N
        C[i, i] = main
    end

    # Set the lower diagonal (wraps around to the top row)
    for i in 2:N
        C[i, i - 1] = lower
    end
    C[1, N] = lower  # Wrap around element for the lower diagonal

    # Set the upper diagonal (wraps around to the bottom row)
    for i in 1:(N - 1)
        C[i, i + 1] = upper
    end
    C[N, 1] = upper  # Wrap around element for the upper diagonal

    return C
end

N = 1000
main = 1
lower = 1
upper = 1

# Example usage
J = tridiagonal_circulant(N, main, lower, upper)
singular_values = svd(J).S

println("Singular values:\n", singular_values)
