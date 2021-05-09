### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 2038c102-6177-11eb-047e-95829eaf59b7
using LinearAlgebra 

# ╔═╡ 924ebce0-6f07-11eb-3b51-a9cdd4ffdd70
using Statistics: mean

# ╔═╡ 8625e7ba-6177-11eb-0f73-f14acfe6a2ea
using Test

# ╔═╡ c1c7bd02-6177-11eb-3073-d92dde78c4bb
using Plots: scatter, plot, histogram, plot!, histogram!, scatter!

# ╔═╡ d2cb6eda-6179-11eb-3e11-41b238bd071d
using Zygote: gradient

# ╔═╡ 87ad3e50-6e7c-11eb-04a4-6b539ab5b671
using Logging # Print training progress to REPL, not pdf

# ╔═╡ 3b614428-6176-11eb-27af-1d611c78a404
md"""
# Regression [30pts]
"""

# ╔═╡ 0abfdd20-7184-11eb-1603-8d3cb40c827b
md"

### I discussed Question 3 with Ben Prystawski and Huifeng Wu.
"

# ╔═╡ 4c8b7872-6176-11eb-3c4c-117dfe425f8a
md"""
### Manually Derived Linear Regression

Suppose that 
$X \in \mathbb{R}^{m \times n}$ with $n \geq m$ 
and $Y \in \mathbb{R}^n$, and that $Y \sim \mathcal{N}(X^T\beta, \sigma^2 I)$.

In this question you will derive the result that the maximum likelihood estimate $\hat\beta$ of $\beta$ is given by

$$\hat\beta = (XX^T)^{-1}XY$$

1. What happens if $n < m$? 

Answer: $XX^T$ is not invertible because rank($X$) = rank($X^T$) = rank($XX^T$) = $n < m$.

2. What are the expectation and covariance matrix of $\hat\beta$, for a given true value of $\beta$? 

Answer: $E(\hat \beta) = E((XX^T)^{-1}XY) = (XX^T)^{-1}XE(Y) = (XX^T)^{-1}(XX^T)\beta = \beta$

$Var(\hat \beta) = E((\hat \beta - \beta)(\hat \beta - \beta)^T) = E((XX^T)^{-1}X (Y - \hat Y)((Y - \hat Y)^T X(XX^T)^{-1}))$

$ = (XX^T)^{-1}X E((Y - \hat Y)(Y - \hat Y)^T) X^T(XX^T)^{-1} = \sigma^2 (XX^T)^{-1} (XX^T)(XX^T)^{-1}$

$= \sigma^2 (XX^T)^{-1}$


3. Show that maximizing the likelihood is equivalent to minimizing the squared error $\sum_{i=1}^n (y_i - x_i\beta)^2$. [Hint: Use $\sum_{i=1}^n a_i^2 = a^Ta$]

Answer: The likelihood function $L(Y) = \frac{1}{\sqrt{(2 \pi)^n \vert \sigma^2 I \vert}} exp(-\frac{1}{2}(Y - X^T \beta)^T(\sigma^2 I)^{-1}（Y - X^T\beta)$

By basic properties of identity matrix $(\sigma^2 I)^{-1} = \frac{1}{\sigma^2} I$

Then $L(Y) = \frac{1}{\sqrt{(2 \pi)^n \vert \sigma^2 I \vert}} exp(-\frac{1}{2 \sigma^2}(Y - X^T \beta)^T(Y - X^T\beta))$

By hint, $(Y - X^T \beta)^T（Y - X^T\beta) = \sum_{i=1}^n (y_i - x_i\beta)^2$.

Then the log-likelihood $l(Y) = log(\frac{1}{\sqrt{(2 \pi)^n \vert \sigma^2 I \vert}}) -\frac{1}{2 \sigma^2}\sum_{i=1}^n (y_i - x_i\beta)^2$ where the 
$1^{st}$ term is constant. 

Essentially we are maximizing over $-\frac{1}{2 \sigma^2}\sum_{i=1}^n (y_i - x_i\beta)^2$, which is equivalent to minimizing $\frac{1}{2 \sigma^2}\sum_{i=1}^n (y_i - x_i\beta)^2$. We can drop the constant term $\frac{1}{2 \sigma^2}$ as it does not affect optimization. Then we obtain that maximizing likelihood is equivalent to minimizing the squared error.

4. Write the squared error in vector notation, (see above hint), expand the expression, and collect like terms. [Hint: Use $\beta^Tx^Ty = y^Tx\beta$ and $x^Tx$ is symmetric]

Answer: $(Y - X^T \beta)^{T}（Y - X^T\beta) = Y^{T}Y - \beta^{T}XY - Y^{T}X^{T}\beta + \beta^{T}XX^{T}\beta$

By hint $\beta^{T}XY = Y^{T}X^{T}\beta$, so the error term equals $Y^{T}Y - 2Y^{T}X^{T}\beta + (X^{T}\beta)^{T}(X^{T}\beta)$



5. Use the likelihood expression to write the negative log-likelihood.
    Write the derivative of the negative log-likelihood with respect to $\beta$, set equal to zero, and solve to show the maximum likelihood estimate $\hat\beta$ as above. 

Answer: negative log-likelihood $-l(Y) = -log(\frac{1}{\sqrt{(2 \pi)^n \vert \sigma^2 I \vert}}) +\frac{1}{2 \sigma^2}(Y^{T}Y - 2Y^{T}X^{T}\beta + (X^{T}\beta)^{T}(X^{T}\beta))$ 

$\frac{\partial -l(Y)}{\partial \beta} = \frac{1}{2 \sigma^2}((-2Y^{T}X^{T})^T + ((X^{T}\beta)^{T}X^{T})^{T} + (X^T)^{T}X^{T}\beta) = 0$

$\Rightarrow \frac{1}{2 \sigma^2} (-2XY + (XX^{T})^{T}\beta + XX^{T}\beta) = 0$ and $XX^{T} = (XX^{T})^{T}$ due to symmetricity

$\Rightarrow -2XY + 2XX^{T}\beta = 0$

$\Rightarrow XX^{T}\beta = XY$

$\Rightarrow \hat \beta = (XX^{T})^{-1}XY$

$\frac{\partial^2 -l(Y)}{\partial \beta^2} = \frac{1}{2 \sigma^2} 2XX^{T}$

Since $XX^{T}$ is always positive semidefinite, we can confirm that the minimum negative log-likelihood is reached. 

QED

"""

# ╔═╡ 16c9fa00-6177-11eb-27b4-175a938132af
md"""
### Toy Data

For visualization purposes and to minimize computational resources we will work with 1-dimensional toy data. 

That is $X \in \mathbb{R}^{m \times n}$ where $m=1$.

We will learn models for 3 target functions

* `target_f1`, linear trend with constant noise.
* `target_f2`, linear trend with heteroskedastic noise.
* `target_f3`, non-linear trend with heteroskedastic noise.
"""

# ╔═╡ 361f0788-6177-11eb-35a9-d12176832720
function target_f1(x, σ_true=0.3)
  noise = randn(size(x))
  y = 2x .+ σ_true.*noise
  return vec(y)
end

# ╔═╡ 361f450e-6177-11eb-1320-9b16d3cbe14c
function target_f2(x)
  noise = randn(size(x))
  y = 2x + norm.(x)*0.3.*noise
  return vec(y)
end

# ╔═╡ 361f9f40-6177-11eb-0792-3f7214a6dafe
function target_f3(x)
  noise = randn(size(x))
  y = 2x + 5sin.(0.5*x) + norm.(x)*0.3.*noise
  return vec(y)
end

# ╔═╡ 61d33d5e-6177-11eb-17be-ff5d831900fa
md"""
### Sample data from the target functions 

Write a function which produces a batch of data $x \sim \text{Uniform}(0,20)$ and `y = target_f(x)`
"""

# ╔═╡ 648df4f8-6177-11eb-225e-8f9955713b67
function sample_batch(target_f, batch_size)
  x = reshape(rand(batch_size) * 20, (1, batch_size))
  y = target_f(x)
  return (x,y)
end

# ╔═╡ 6ac7bb10-6177-11eb-02bc-3513af63a9b9
md""" 
### Test assumptions about your dimensions
"""

# ╔═╡ 8c5b76a4-6177-11eb-223a-6b656e173ffb
let
@testset "sample dimensions are correct" begin
  m = 1 # dimensionality
  n = 200 # batch-size
  for target_f in (target_f1, target_f2, target_f3)
    x,y = sample_batch(target_f,n)
    @test size(x) == (m,n)
    @test size(y) == (n,)
  end
end
end

# ╔═╡ 91b725a8-6177-11eb-133f-cb075b2c50dc
md"""
### Plot the target functions

For all three targets, plot a $n=1000$ sample of the data.

**Note: You will use these plots later, in your writeup.
Conmsider suppressing display once other questions are complete.**
"""

# ╔═╡ f6999bb8-6177-11eb-3a0e-35e7a718e0ab
md"""## Linear Regression Model with $\hat \beta$ MLE"""

# ╔═╡ f9f79404-6177-11eb-2d31-857cb99cf7db
md"""
### Code the hand-derived MLE

Program the function that computes the the maximum likelihood estimate given $X$ and $Y$.
    Use it to compute the estimate $\hat \beta$ for a $n=1000$ sample from each target function.
"""

# ╔═╡ 1cfe23f0-6178-11eb-21bf-8d8a46b0d7c6
function beta_mle(X,Y)
  beta = inv(X * X') * X * Y#TODO
  return beta
end

# ╔═╡ c8b5caf0-6ee7-11eb-313b-4f42d84c1b84
function mle_plot(target_f, mle=false)
	x, y = sample_batch(target_f, 1000)
	
	plot_mle = plot(fmt=:png)
	
	scatter!(reshape(x, (1000,)), y, label="$target_f _sample")
	
	if mle == true
		β_mle = beta_mle(x, y) #TODO
		x_line = 0:0.01:20
	plot!(x_line, reshape(x_line, (1, 2001))' * β_mle, ribbon=1, label="$target_f _mle")
	end
	return plot_mle
end

# ╔═╡ 2be53474-6178-11eb-2b75-1f1c3187770f
mle_plot(target_f1) #TODO

# ╔═╡ 2be5b9b4-6178-11eb-3e01-61fea97cdfa6
mle_plot(target_f2)#TODO

# ╔═╡ 2bffc028-6178-11eb-0718-199b410131eb
mle_plot(target_f3)

# ╔═╡ 242049d0-6178-11eb-2572-ad3f6ea43127
let 
	x, y = sample_batch(target_f1, 1000)
	β_mle_1 = beta_mle(x, y)
end

# ╔═╡ 242af220-6178-11eb-3bfe-6bcf3e814ff1
let 
	x, y = sample_batch(target_f2, 1000)
	β_mle_2 = beta_mle(x, y)
end

# ╔═╡ 24329fca-6178-11eb-0bdc-45e28e499635
let 
	x, y = sample_batch(target_f3, 1000)
	β_mle_3 = beta_mle(x, y)
end

# ╔═╡ 4f04aca2-6178-11eb-0e59-df88659f11c1
md"""
### Plot the MLE linear regression model

For each function, plot the linear regression model given by $Y \sim \mathcal{N}(X^T\hat\beta, \sigma^2 I)$ for $\sigma=1.$.
    This plot should have the line of best fit given by the maximum likelihood estimate, as well as a shaded region around the line corresponding to plus/minus one standard deviation (i.e. the fixed uncertainty $\sigma=1.0$).
    Using `Plots.jl` this shaded uncertainty region can be achieved with the `ribbon` keyword argument.
    **Display 3 plots, one for each target function, showing samples of data and maximum likelihood estimate linear regression model**
"""

# ╔═╡ 6222e9e0-6178-11eb-3448-57e951600ef9
mle_plot(target_f1, true) #TODO

# ╔═╡ 705a7d28-6178-11eb-280d-315614ad9080
mle_plot(target_f2, true) #TODO

# ╔═╡ 742afde4-6178-11eb-08b6-e77847b0aa40
mle_plot(target_f3, true) #TODO

# ╔═╡ 7f61ff44-6178-11eb-040b-2f19e52ebaf5
md"""
## Log-likelihood of Data Under Model

### Code for Gaussian Log-Likelihood

Write code for the function that computes the likelihood of $x$ under the Gaussian distribution $\mathcal{N}(μ,σ)$.
For reasons that will be clear later, this function should be able to broadcast to the case where $x, \mu, \sigma$ are all vector valued
and return a vector of likelihoods with equivalent length, i.e., $x_i \sim \mathcal{N}(\mu_i,\sigma_i)$.
"""

# ╔═╡ baf4cbea-6178-11eb-247f-4961c91da925
function gaussian_log_likelihood(μ, σ, x)
  """
  compute log-likelihood of x under N(μ,σ)
  """
  constant_term = -1/2 * (log(2*pi) + 2 * log(σ))
  exp_term = -1/(2σ^2) * (x - μ)^2
  return constant_term + exp_term 
end

# ╔═╡ e8da3e14-6178-11eb-2058-6d5f4a77378b
let
@testset "Gaussian log likelihood" begin
using Distributions: pdf, Normal
# Scalar mean and variance
x = randn()
μ = randn()
σ = rand()
@test size(gaussian_log_likelihood(μ,σ,x)) == () # Scalar log-likelihood
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x)) # Correct Value
# Vector valued x under constant mean and variance
x = randn(100)
μ = randn()
σ = rand()
@test size(gaussian_log_likelihood.(μ,σ,x)) == (100,) # Vector of log-likelihoods
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x)) # Correct Values
# Vector valued x under vector valued mean and variance
x = randn(10)
μ = randn(10)
σ = rand(10)
@test size(gaussian_log_likelihood.(μ,σ,x)) == (10,) # Vector of log-likelihoods
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal.(μ,σ),x)) # Correct Values
end
end

# ╔═╡ ccedbd3e-6178-11eb-2d26-01016c9dea4b
md"""
### Test Gaussian likelihood against standard implementation
"""

# ╔═╡ 045fd22a-6179-11eb-3901-8b327ed3b7a0
md"""
### Model Negative Log-Likelihood

Use your gaussian log-likelihood function to write the code which computes the negative log-likelihood of the target value $Y$ under the model $Y \sim \mathcal{N}(X^T\beta, \sigma^2*I)$ for 
a given value of $\beta$.
"""

# ╔═╡ 2b7be59c-6179-11eb-3fde-01eac8b60a9d
function lr_model_nll(β,x,y;σ=1.)
  return -sum(gaussian_log_likelihood.(x' * β, σ, y)) #TODO: Negative Log Likelihood
end

# ╔═╡ 314fd0fa-6179-11eb-0d0c-a7efdfa33c65
md"""
### Compute Negative-Log-Likelihood on data

Use this function to compute and report the negative-log-likelihood of a $n\in \{10,100,1000\}$ batch of data
under the model with the maximum-likelihood estimate $\hat\beta$ and $\sigma \in \{0.1,0.3,1.,2.\}$ for each target function.
"""

# ╔═╡ 5b1dec3c-6179-11eb-2968-d361e1c18cc8
for n in (10, 100, 1000)
    println("--------  $n  ------------")
    for target_f in (target_f1,target_f2, target_f3)
      println("--------  $target_f  ------------")
      for σ_model in (0.1,0.3,1.,2.)
        println("--------  $σ_model  ------------")
        x,y = sample_batch(target_f, n)#TODO
        β_mle = beta_mle(x, y)#TODO
        nll = lr_model_nll(β_mle, x, y; σ=σ_model)#TODO
        println("Negative Log-Likelihood: $nll")
      end
    end
end

# ╔═╡ 6cdd6d12-6179-11eb-1115-cd43e97e9a60
md"""
### Effect of model variance

For each target function, what is the best choice of $\sigma$? 


Please note that $\sigma$ and batch-size $n$ are modelling hyperparameters. 
In the expression of maximum likelihood estimate, $\sigma$ or $n$ do not appear, and in principle shouldn't affect the final answer.
However, in practice these can have significant effect on the numerical stability of the model. 
Too small values of $\sigma$ will make data away from the mean very unlikely, which can cause issues with precision.
Also, the negative log-likelihood objective involves a sum over the log-likelihoods of each datapoint. This means that with a larger batch-size $n$, there are more datapoints to sum over, so a larger negative log-likelihood is not necessarily worse. 
The take-home is that you cannot directly compare the negative log-likelihoods achieved by these models with different hyperparameter settings.

Answer: 

For $target\_f1$, the best $\sigma$ is 0.3, which is the true $\sigma$ parameter.

For $target\_f2$ and $target\_f3$ the best $\sigma$ is 2.0. The reason for a larger $\sigma$ is due to the heteroskedastic data and non-linearity. The negative log-likelihood for $target\_f3$ is higher than that for $target\_f3$ given the same $n$ and $\sigma$, showing that non-linearity further worsens the estimation.


"""

# 10        100         1000
# target_f1 0.3/3.81, 0.3/28.56,  0.3/232.31
# target_f2 2.0/18.83, 2.0/342.50, 2.0/3096.45
# target_f3 2.0/54.41, 2.0/415.04, 2.0/4507.22

# ╔═╡ 97606f8a-6179-11eb-3dbb-af72ec35e4a1
md"""
## Automatic Differentiation and Maximizing Likelihood

In a previous question you derived the expression for the derivative of the negative log-likelihood with respect to $\beta$.
We will use that to test the gradients produced by automatic differentiation.
"""

# ╔═╡ abe7efdc-6179-11eb-14ed-a3b0462cc2f0
md"""
### Compute Gradients with AD, Test against hand-derived
For a random value of $\beta$, $\sigma$, and $n=100$ sample from a target function,
    use automatic differentiation to compute the derivative of the negative log-likelihood of the sampled data
with respect $\beta$.
Test that this is equivalent to the hand-derived value.
"""

# ╔═╡ d4a5a4e6-6179-11eb-0596-bb8ddc6027fb
@testset "Gradients wrt parameter" begin
	β_test = randn()
	σ_test = rand()
	x,y = sample_batch(target_f1,100)
	ad_grad = gradient(β -> lr_model_nll(β, x, y; σ=σ_test), β_test)
	hand_derivative = (1/σ_test^2 * (-x * y + x * x' * β_test))[1]
@test ad_grad[1] ≈ hand_derivative
end

# ╔═╡ d9d4d6da-6179-11eb-0165-95d79e1ab92d
md"""
### Train Linear Regression Model with Gradient Descent

In this question we will compute gradients of of negative log-likelihood with respect to $\beta$.
We will use gradient descent to find $\beta$ that maximizes the likelihood.

Write a function `train_lin_reg` that accepts a target function and an initial estimate for $\beta$ and some 
hyperparameters for batch-size, model variance, learning rate, and number of iterations.

Then, for each iteration:
* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\beta$
* update the estimate of $\beta$ with gradient descent with specified learning rate
and, after all iterations, returns the final estimate of $\beta$.
"""

# ╔═╡ 033afc0c-617a-11eb-32a9-f3f467476a0a
function train_lin_reg(target_f, β_init; bs= 100, lr = 1e-6, iters=1000, σ_model = 1.)
    β_curr = β_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)#TODO
      @info "loss: $β_init β: $β_curr" #TODO: log loss, if you want to monitor training progress
      grad_β = (1/σ_model^2 * (-x * y + x * x' * β_curr))[1]#TODO: compute gradients
      β_curr = β_curr - lr * grad_β#TODO: gradient descent
    end
    return β_curr
end

# ╔═╡ 217b1466-617a-11eb-19c6-2f29ef5d3576
md"""
### Parameter estimate by gradient descent

For each target function, start with an initial parameter $\beta$, 
    learn an estimate for $\beta_\text{learned}$ by gradient descent.
    Then plot a $n=1000$ sample of the data and the learned linear regression model with shaded region for uncertainty corresponding to plus/minus one standard deviation.
"""

# ╔═╡ 449d93e6-617a-11eb-3289-a931618f4bba
function plot_grad_descent(target_f)
	plot_l = plot()
	β_init = 1000*randn() # Initial parameter
	β_learned = train_lin_reg(target_f, β_init; bs= 100, lr = 1e-6, iters=1000, σ_model = 1.)
	sample_x, sample_y = sample_batch(target_f, 1000)
	scatter!(reshape(sample_x, (1000,)), sample_y, fmt=:png, label="$target_f _sample")
	x_line = 0:0.01:20
	plot!(x_line, reshape(x_line, (1, 2001))' * β_learned, ribbon=1, label="$target_f grad_desc")
	return plot_l
end
	

# ╔═╡ 4adae2ac-617a-11eb-0ed7-cdb1985fad44
plot_grad_descent(target_f1)

# ╔═╡ 8a34de20-6e7d-11eb-1aff-fb3e71b058b0
plot_grad_descent(target_f2)

# ╔═╡ d5f8c5b0-6e7d-11eb-0273-1b8e9a814c82
plot_grad_descent(target_f3)

# ╔═╡ 78cc51f0-617a-11eb-3408-83b8d44df832
md"""
### Non-linear Regression with a Neural Network

In the previous questions we have considered a linear regression model 

$$Y \sim \mathcal{N}(X^T \beta, \sigma^2)$$

This model specified the mean of the predictive distribution for each datapoint by the product of that datapoint with our parameter.

Now, let us generalize this to consider a model where the mean of the predictive distribution is a non-linear function of each datapoint.
We will have our non-linear model be a simple function called `neural_net` with parameters $\theta$ 
(collection of weights and biases).

$$Y \sim \mathcal{N}(\texttt{neural\_net}(X,\theta), \sigma^2)$$
"""

# ╔═╡ 8d762018-617a-11eb-3def-27d3959bb155
md"""
### Fully-connected Neural Network

Write the code for a fully-connected neural network (multi-layer perceptron) with one 10-dimensional hidden layer and a `tanh` nonlinearirty.
You must write this yourself using only basic operations like matrix multiply and `tanh`, you may not use layers provided by a library.

This network will output the mean vector, test that it outputs the correct shape for some random parameters.
"""

# ╔═╡ a95e69f2-617a-11eb-3034-e12b72492357
begin
# Neural Network Function

function neural_net(x, θ)
		
	# Dimensionality
	# X: m x n, W1: m x 10, b1: 10 x 1, h1: 10 x n, 
	# W2: 10 x 1, b2: 1 x 1, y: n x 1
		
	z1 = affine_forward(x, θ.w1, θ.b1)
	h1 = tanh_forward(z1)
	y = vec(affine_forward(h1, θ.w2, θ.b2)')
	return y
end
	
function affine_forward(x, w, b)
	return w' * x .+ b
end
	
function tanh_forward(x)
	return tanh.(x)
end
	
end

# ╔═╡ 99d1a54e-6eed-11eb-146c-2356a5c37455
function rand_θ(input_dim=1, output_dim=1, hidden_dim=10)
	coef = 0.01
	θ_neural_net = (w1 = coef * randn(input_dim, hidden_dim), 
	  				b1 = coef * randn(hidden_dim, 1), 
	 				w2 = coef * randn(hidden_dim, output_dim),
	  				b2 = coef * randn(output_dim, 1))
	return θ_neural_net
end

# ╔═╡ 6d8ce550-6e88-11eb-33cc-076a237cb714
θ_nn_reg = rand_θ()

# ╔═╡ ca10a7fa-617a-11eb-1568-4724c3686b01
md"""
### Test assumptions about model output

Test, at least, the dimension assumptions.
"""

# ╔═╡ bfa935e8-617a-11eb-364c-93b89a9b3e23
let
@testset "neural net mean vector output" begin
n = 100
x,y = sample_batch(target_f1,n)
μ = neural_net(x,θ_nn_reg)
@test size(μ) == (n,)
end
end

# ╔═╡ e3a14bac-617a-11eb-2155-ff812544df13
md"""
### Negative Log-likelihood of NN model
Write the code that computes the negative log-likelihood for this model where the mean is given by the output of the neural network and $\sigma = 1.0$
"""

# ╔═╡ f65af5b8-617a-11eb-3865-7fa972a6a821
function nn_model_nll(θ,x,y;σ=1)
  return -sum(gaussian_log_likelihood.(neural_net(x, θ), σ, y))
end

# ╔═╡ 0f355c54-617b-11eb-2768-7b8066538440
md"""
### Training model to maximize likelihood

Write a function `train_nn_reg` that accepts a target function and an initial estimate for $\theta$ and some 
    hyperparameters for batch-size, model variance, learning rate, and number of iterations.

Then, for each iteration:
* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\theta$
* update the estimate of $\theta$ with gradient descent with specified learning rate
and, after all iterations, returns the final estimate of $\theta$.
"""

# ╔═╡ 3ca4bbbc-617b-11eb-36e4-ad24b86bfd19
begin
function train_nn_reg(target_f, θ_init; bs= 100, lr = 1e-5, iters=1000, σ_model = 1. )
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)
	  log_loss = nn_model_nll(θ_init, x, y; σ=σ_model)
      @info "loss: $log_loss" 
      grad_θ = gradient((θ -> nn_model_nll(θ, x, y; σ=σ_model)), θ_curr)[1]
      θ_curr = nn_descent(θ_curr, lr, grad_θ)
    end
    return θ_curr
end

	
function nn_descent(θ, lr, grad)
	θ.w1 .= θ.w1 - lr * grad.w1
	θ.b1 .= θ.b1 - lr * grad.b1
	θ.w2 .= θ.w2 - lr * grad.w2
	θ.b2 .= θ.b2 - lr * grad.b2
	return θ
end
end

# ╔═╡ 429d9e76-617b-11eb-161c-2b16653d2b0c
md"""
### Learn model parameters

For each target function, start with an initialization of the network parameters, $\theta$,
    use your train function to minimize the negative log-likelihood and find an estimate for $\theta_\text{learned}$ by gradient descent.
    
"""

# ╔═╡ 76a2ec3e-6eef-11eb-12a8-cf709c0127e1
begin
	θ_nn_f1 = train_nn_reg(target_f1, rand_θ(); bs=1000, lr=1e-5, iters=25000)
	θ_nn_f2 = train_nn_reg(target_f2, rand_θ(); bs=1000, lr=1e-5, iters=25000)
	θ_nn_f3 = train_nn_reg(target_f3, rand_θ(); bs=1000, lr=1e-5, iters=25000)
end

# ╔═╡ 5b9a5c98-617b-11eb-2ede-9dbbeb8ea32d
md"""
### Plot neural network regression

Then plot a $n=1000$ sample of the data and the learned regression model with shaded uncertainty bounds given by $\sigma = 1.0$
"""

# ╔═╡ 654627c2-6e8e-11eb-0157-c5290b30cbd1
function plot_neural_net(target_f, θ_trained)
	plot_nn = plot(fmt=:png)
	sample_x, sample_y = sample_batch(target_f, 1000)
	scatter!(reshape(sample_x, (1000,)), sample_y, label="$target_f")
	x_line = 0:0.01:20
	plot!(x_line, neural_net(x_line', θ_trained), ribbon=1, label="$target_f _nn_reg")
	return plot_nn
end

# ╔═╡ 65bf4f60-6e8e-11eb-2f7c-a750d141b5e3
plot_neural_net(target_f1, θ_nn_f1)

# ╔═╡ 3394a392-6e8f-11eb-3608-558f0d2653d9
plot_neural_net(target_f2, θ_nn_f2)

# ╔═╡ 34198b00-6e8f-11eb-2e7b-378c22070113
plot_neural_net(target_f3, θ_nn_f3)

# ╔═╡ 89cdc082-617b-11eb-3491-453302b03caa
md"""
## Input-dependent Variance

In the previous questions we've gone from a gaussian model with mean given by linear combination

$$Y \sim \mathcal{N}(X^T \beta, \sigma^2)$$

to gaussian model with mean given by non-linear function of the data (neural network)

$$Y \sim \mathcal{N}(\texttt{neural\_net}(X,\theta), \sigma^2)$$

However, in all cases we have considered so far, we specify a fixed variance for our model distribution.
We know that two of our target datasets have heteroscedastic noise, meaning any fixed choice of variance will poorly model the data.

In this question we will use a neural network to learn both the mean and log-variance of our gaussian model.

$$\begin{align*}
\mu, \log \sigma &= \texttt{neural\_net}(X,\theta)\\
Y &\sim \mathcal{N}(\mu, \exp(\log \sigma)^2)
\end{align*}$$
"""

# ╔═╡ b04e11d0-617b-11eb-35a4-b5aeeafb8570
md"""
### Neural Network that outputs log-variance

Write the code for a fully-connected neural network (multi-layer perceptron) with one 10-dimensional hidden layer and a `tanh` nonlinearirty, and outputs both a vector for mean and $\log \sigma$. 
"""

# ╔═╡ c7ae64a6-617b-11eb-10b8-712abc9dc2a6
# Neural Network with variance
function neural_net_w_var(x,θ)
	y = neural_net(x, θ.mu)
	logσ = neural_net(x, θ.sigma)
  return y, logσ
end

# ╔═╡ d6ad1a06-617b-11eb-224c-0307a6e6f80a
# Random initial Parameters
function rand_θ_w_var(input_dim=1, output_dim=1, hidden_dim=10)
	θ_nn_w_var = (mu = rand_θ(input_dim, output_dim, hidden_dim),
				  sigma = rand_θ(input_dim, output_dim, hidden_dim))
	return θ_nn_w_var
end

# ╔═╡ c82cff46-617b-11eb-108a-a9576c728328
md"""
### Test model assumptions

Test the output shape is as expected.
"""

# ╔═╡ d9c41410-617b-11eb-2542-6da3f6bd0498
let
@testset "neural net mean and logsigma vector output" begin
n = 10000
x,y = sample_batch(target_f1,n)
μ, logσ = neural_net_w_var(x,rand_θ_w_var())
@test size(μ) == (n,)
@test size(logσ) == (n,)
end
end

# ╔═╡ e7cddc4e-617b-11eb-3b2c-f7c06fc61fd5
md"""
### Negative log-likelihood with modelled variance

Write the code that computes the negative log-likelihood for this model where the mean and $\log \sigma$ is given by the output of the neural network.
    
(Hint: Don't forget to take $\exp \log \sigma$)
"""

# ╔═╡ 0b7d0ac0-617c-11eb-24c5-b3126ee28f5a
function nn_with_var_model_nll(θ,x,y)
  mu, sigma = neural_net_w_var(x, θ)
  return -sum(gaussian_log_likelihood.(mu, exp.(sigma), y))
end

# ╔═╡ 128daf4a-617c-11eb-3c62-1b61708169e0
md"""
### Write training loop

Write a function `train_nn_w_var_reg` that accepts a target function and an initial estimate for $\theta$ and some 
    hyperparameters for batch-size, learning rate, and number of iterations.

Then, for each iteration:

* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\theta$
* update the estimate of $\theta$ with gradient descent with specified learning rate

and, after all iterations, returns the final estimate of $\theta$.
"""

# ╔═╡ 3c657688-617c-11eb-2655-415562d132bb
function train_nn_w_var_reg(target_f, θ_init; bs= 100, lr = 1e-4, iters=10000)
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)#TODO
	  log_loss = nn_with_var_model_nll(θ_curr, x, y)
      @info "loss: $log_loss" #TODO: log loss
      grad_θ = gradient((θ -> nn_with_var_model_nll(θ, x, y)), θ_curr)[1]
	  θ_curr = (mu = nn_descent(θ_curr.mu, lr, grad_θ.mu),
				sigma = nn_descent(θ_curr.sigma, lr, grad_θ.sigma))
    end
    return θ_curr
end

# ╔═╡ 44cdb444-617c-11eb-1c69-8bb0197c9c32
md"""
### Learn model with input-dependent variance

 For each target function, start with an initialization of the network parameters, $\theta$,
    learn an estimate for $\theta_\text{learned}$ by gradient descent.
    Then plot a $n=1000$ sample of the dataset and the learned regression model with shaded uncertainty bounds corresponding to plus/minus one standard deviation given by the variance of the predictive distribution at each input location 
    (output by the neural network).
    (Hint: `ribbon` argument for shaded uncertainty bounds can accept a vector of $\sigma$)

Note: Learning the variance is tricky, and this may be unstable during training. There are some things you can try:
* Adjusting the hyperparameters like learning rate and batch size
* Train for more iterations
* Try a different random initialization, like sample random weights and bias matrices with lower variance.
    
For this question **you will not be assessed on the final quality of your model**.
Specifically, if you fails to train an optimal model for the data that is okay. 
You are expected to learn something that is somewhat reasonable, and **demonstrates that this model is training and learning variance**.

If your implementation is correct, it is possible to learn a reasonable model with fewer than 10 minutes of training on a laptop CPU. 
The default hyperparameters should help, but may need some tuning.

"""

# ╔═╡ 74bd837a-617c-11eb-3716-07cd84f5f4ac
#TODO: For each target function
begin
	θ_nn_f1_w_var = train_nn_w_var_reg(target_f1, rand_θ_w_var(); bs=1000, lr=1e-5, iters=25000)
	θ_nn_f2_w_var = train_nn_w_var_reg(target_f2, rand_θ_w_var(); bs=1000, lr=1e-5, iters=25000)
	θ_nn_f3_w_var = train_nn_w_var_reg(target_f3, rand_θ_w_var(); bs=1000, lr=1e-5, iters=50000)
end

# ╔═╡ 79674636-617c-11eb-0213-fb99e78e9f1d
md"""
### Plot model
"""

# ╔═╡ e5fb5a90-6ea0-11eb-24ee-19d13bd0e193
function plot_neural_net_w_var(target_f, θ_trained)
	plot_nn = plot(fmt=:png)
	sample_x, sample_y = sample_batch(target_f, 1000)
	scatter!(reshape(sample_x, (1000,)), sample_y, label="$target_f _sample")
	x_line = 0:0.01:20
	mu, sigma = neural_net_w_var(x_line', θ_trained)
	plot!(x_line, mu, ribbon=exp.(sigma), label="$target_f _nn_reg_w_var ")
	return plot_nn
end

# ╔═╡ e5fc9310-6ea0-11eb-3ae5-d3704e3018ad
plot_neural_net_w_var(target_f1, θ_nn_f1_w_var)

# ╔═╡ e5fa9740-6ea0-11eb-23c9-832717e87c63
plot_neural_net_w_var(target_f2, θ_nn_f2_w_var)

# ╔═╡ e5f9ace0-6ea0-11eb-010f-91331637006c
plot_neural_net_w_var(target_f3, θ_nn_f3_w_var)

# ╔═╡ 8c3f1d38-617c-11eb-2820-f32c96e276c6
md"""
### Spend time making it better (optional)

If you would like to take the time to train a very good model of the data (specifically for target functions 2 and 3) with a neural network
that outputs both mean and $\log \sigma$ you can do this, but it is not necessary to achieve full marks.

You can try
* Using a more stable optimizer, like Adam. You may import this from a library.
* Increasing the expressivity of the neural network, increase the number of layers or the dimensionality of the hidden layer.
* Careful tuning of hyperparameters, like learning rate and batchsize.

Answer: Added another hidden layer of dimension 10
"""

# ╔═╡ 70a023e0-6ef8-11eb-16ec-8fa3011d3242
begin
	
function rand_θ_extra(input_dim=1, output_dim=1, hidden_dim_1=10, hidden_dim_2=10)
	coef = 0.01
	θ_neural_net = (w1 = coef * randn(input_dim, hidden_dim_1), 
	  b1 = coef * randn(hidden_dim_1, 1), 
	  w2 = coef * randn(hidden_dim_1, hidden_dim_2),
	  b2 = coef * randn(hidden_dim_2, 1),
	  w3 = coef * randn(hidden_dim_2, output_dim),
	  b3 = coef * randn(output_dim, 1))
	return θ_neural_net
end

function rand_θ_w_var_extra(input_dim=1, output_dim=1, hidden_dim_1=10, hidden_dim_2=10)
	θ_nn_w_var = (mu = rand_θ_extra(input_dim, output_dim, hidden_dim_1, hidden_dim_2),
				  sigma = rand_θ_extra(input_dim, output_dim, hidden_dim_1, hidden_dim_2))
	return θ_nn_w_var
end
	
end

# ╔═╡ 7605da32-6ef5-11eb-13cb-ad7270aa4995
begin
# Neural Network Function

function neural_net_extra(x, θ)
		
	# Dimensionality
	# X: m x n, W1: m x 10, b1: 10 x 1, h1: 10 x n, 
	# W2: 10 x 1, b2: 1 x 1, y: n x 1
		
	z1 = affine_forward(x, θ.w1, θ.b1)
	h1 = tanh_forward(z1)
	z2 = affine_forward(h1, θ.w2, θ.b2)
	h2 = tanh_forward(z2)
	y = vec(affine_forward(h2, θ.w3, θ.b3)')
	return y
end
	
# Neural Network with variance
function neural_net_w_var_extra(x,θ)
	y = neural_net_extra(x, θ.mu)
	logσ = neural_net_extra(x, θ.sigma)
  return y, logσ
end
	

function nn_descent_extra(θ, lr, grad)
	θ.w1 .= θ.w1 - lr * grad.w1
	θ.b1 .= θ.b1 - lr * grad.b1
	θ.w2 .= θ.w2 - lr * grad.w2
	θ.b2 .= θ.b2 - lr * grad.b2
	θ.w3 .= θ.w3 - lr * grad.w3
	θ.b3 .= θ.b3 - lr * grad.b3
	return θ
end
	
end

# ╔═╡ 9acfda70-6ef8-11eb-0800-d1f21da4c85c
begin
function nn_with_var_model_nll_extra(θ,x,y)
  mu, sigma = neural_net_w_var_extra(x, θ)
  return -sum(gaussian_log_likelihood.(mu, exp.(sigma), y))
end
	
function train_nn_w_var_reg_extra(target_f, θ_init; bs= 100, lr = 1e-4, iters=10000)
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)#TODO
	  log_loss = nn_with_var_model_nll_extra(θ_curr, x, y)
      @info "loss: $log_loss" #TODO: log loss
      grad_θ = gradient((θ -> nn_with_var_model_nll_extra(θ, x, y)), θ_curr)[1]
	  θ_curr = (mu = nn_descent_extra(θ_curr.mu, lr, grad_θ.mu),
				sigma = nn_descent_extra(θ_curr.sigma, lr, grad_θ.sigma))
    end
    return θ_curr
end
end

# ╔═╡ eb6abaf0-6ef7-11eb-0cf8-bb56211984c5
#TODO: For each target function
begin
	θ_nn_f1_w_var_extra = train_nn_w_var_reg_extra(target_f1, rand_θ_w_var_extra(); bs=1000, lr=1e-5, iters=75000)
	θ_nn_f2_w_var_extra = train_nn_w_var_reg_extra(target_f2, rand_θ_w_var_extra(); bs=1000, lr=1e-5, iters=75000)
	θ_nn_f3_w_var_extra = train_nn_w_var_reg_extra(target_f3, rand_θ_w_var_extra();
bs=1000, lr=1e-5, iters=75000)
end

# ╔═╡ e643f450-6ef8-11eb-3858-fb9cf3e78cc6
function plot_neural_net_w_var_extra(target_f, θ_trained)
	plot_nn = plot(fmt=:png)
	sample_x, sample_y = sample_batch(target_f, 1000)
	scatter!(reshape(sample_x, (1000,)), sample_y, label="$target_f _sample")
	x_line = 0:0.01:20
	mu, sigma = neural_net_w_var_extra(x_line', θ_trained)
	plot!(x_line, mu, ribbon=exp.(sigma), label="$target_f _nn_reg_w_var_extra ")
	return plot_nn
end

# ╔═╡ 0fe93dd0-6efc-11eb-1366-85e28ccc33fa
plot_neural_net_w_var_extra(target_f1, θ_nn_f1_w_var_extra)

# ╔═╡ 103e3ba0-6efc-11eb-18f0-bd3fb799bf3f
plot_neural_net_w_var_extra(target_f2, θ_nn_f2_w_var_extra)

# ╔═╡ 10bf2b70-6efc-11eb-3c36-cb41af9baffb
plot_neural_net_w_var_extra(target_f3, θ_nn_f3_w_var_extra)

# ╔═╡ Cell order:
# ╟─3b614428-6176-11eb-27af-1d611c78a404
# ╟─0abfdd20-7184-11eb-1603-8d3cb40c827b
# ╟─4c8b7872-6176-11eb-3c4c-117dfe425f8a
# ╟─16c9fa00-6177-11eb-27b4-175a938132af
# ╠═2038c102-6177-11eb-047e-95829eaf59b7
# ╠═924ebce0-6f07-11eb-3b51-a9cdd4ffdd70
# ╠═361f0788-6177-11eb-35a9-d12176832720
# ╠═361f450e-6177-11eb-1320-9b16d3cbe14c
# ╠═361f9f40-6177-11eb-0792-3f7214a6dafe
# ╟─61d33d5e-6177-11eb-17be-ff5d831900fa
# ╠═648df4f8-6177-11eb-225e-8f9955713b67
# ╟─6ac7bb10-6177-11eb-02bc-3513af63a9b9
# ╠═8625e7ba-6177-11eb-0f73-f14acfe6a2ea
# ╠═8c5b76a4-6177-11eb-223a-6b656e173ffb
# ╟─91b725a8-6177-11eb-133f-cb075b2c50dc
# ╠═c1c7bd02-6177-11eb-3073-d92dde78c4bb
# ╠═c8b5caf0-6ee7-11eb-313b-4f42d84c1b84
# ╠═2be53474-6178-11eb-2b75-1f1c3187770f
# ╠═2be5b9b4-6178-11eb-3e01-61fea97cdfa6
# ╠═2bffc028-6178-11eb-0718-199b410131eb
# ╟─f6999bb8-6177-11eb-3a0e-35e7a718e0ab
# ╟─f9f79404-6177-11eb-2d31-857cb99cf7db
# ╠═1cfe23f0-6178-11eb-21bf-8d8a46b0d7c6
# ╠═242049d0-6178-11eb-2572-ad3f6ea43127
# ╠═242af220-6178-11eb-3bfe-6bcf3e814ff1
# ╠═24329fca-6178-11eb-0bdc-45e28e499635
# ╟─4f04aca2-6178-11eb-0e59-df88659f11c1
# ╠═6222e9e0-6178-11eb-3448-57e951600ef9
# ╠═705a7d28-6178-11eb-280d-315614ad9080
# ╠═742afde4-6178-11eb-08b6-e77847b0aa40
# ╟─7f61ff44-6178-11eb-040b-2f19e52ebaf5
# ╠═baf4cbea-6178-11eb-247f-4961c91da925
# ╟─ccedbd3e-6178-11eb-2d26-01016c9dea4b
# ╠═e8da3e14-6178-11eb-2058-6d5f4a77378b
# ╟─045fd22a-6179-11eb-3901-8b327ed3b7a0
# ╠═2b7be59c-6179-11eb-3fde-01eac8b60a9d
# ╟─314fd0fa-6179-11eb-0d0c-a7efdfa33c65
# ╠═5b1dec3c-6179-11eb-2968-d361e1c18cc8
# ╟─6cdd6d12-6179-11eb-1115-cd43e97e9a60
# ╟─97606f8a-6179-11eb-3dbb-af72ec35e4a1
# ╟─abe7efdc-6179-11eb-14ed-a3b0462cc2f0
# ╠═d2cb6eda-6179-11eb-3e11-41b238bd071d
# ╠═d4a5a4e6-6179-11eb-0596-bb8ddc6027fb
# ╟─d9d4d6da-6179-11eb-0165-95d79e1ab92d
# ╠═87ad3e50-6e7c-11eb-04a4-6b539ab5b671
# ╠═033afc0c-617a-11eb-32a9-f3f467476a0a
# ╟─217b1466-617a-11eb-19c6-2f29ef5d3576
# ╠═449d93e6-617a-11eb-3289-a931618f4bba
# ╠═4adae2ac-617a-11eb-0ed7-cdb1985fad44
# ╠═8a34de20-6e7d-11eb-1aff-fb3e71b058b0
# ╠═d5f8c5b0-6e7d-11eb-0273-1b8e9a814c82
# ╟─78cc51f0-617a-11eb-3408-83b8d44df832
# ╟─8d762018-617a-11eb-3def-27d3959bb155
# ╠═a95e69f2-617a-11eb-3034-e12b72492357
# ╠═99d1a54e-6eed-11eb-146c-2356a5c37455
# ╠═6d8ce550-6e88-11eb-33cc-076a237cb714
# ╟─ca10a7fa-617a-11eb-1568-4724c3686b01
# ╠═bfa935e8-617a-11eb-364c-93b89a9b3e23
# ╟─e3a14bac-617a-11eb-2155-ff812544df13
# ╠═f65af5b8-617a-11eb-3865-7fa972a6a821
# ╟─0f355c54-617b-11eb-2768-7b8066538440
# ╠═3ca4bbbc-617b-11eb-36e4-ad24b86bfd19
# ╟─429d9e76-617b-11eb-161c-2b16653d2b0c
# ╠═76a2ec3e-6eef-11eb-12a8-cf709c0127e1
# ╟─5b9a5c98-617b-11eb-2ede-9dbbeb8ea32d
# ╠═654627c2-6e8e-11eb-0157-c5290b30cbd1
# ╠═65bf4f60-6e8e-11eb-2f7c-a750d141b5e3
# ╠═3394a392-6e8f-11eb-3608-558f0d2653d9
# ╠═34198b00-6e8f-11eb-2e7b-378c22070113
# ╟─89cdc082-617b-11eb-3491-453302b03caa
# ╟─b04e11d0-617b-11eb-35a4-b5aeeafb8570
# ╠═c7ae64a6-617b-11eb-10b8-712abc9dc2a6
# ╠═d6ad1a06-617b-11eb-224c-0307a6e6f80a
# ╟─c82cff46-617b-11eb-108a-a9576c728328
# ╠═d9c41410-617b-11eb-2542-6da3f6bd0498
# ╟─e7cddc4e-617b-11eb-3b2c-f7c06fc61fd5
# ╠═0b7d0ac0-617c-11eb-24c5-b3126ee28f5a
# ╟─128daf4a-617c-11eb-3c62-1b61708169e0
# ╠═3c657688-617c-11eb-2655-415562d132bb
# ╟─44cdb444-617c-11eb-1c69-8bb0197c9c32
# ╠═74bd837a-617c-11eb-3716-07cd84f5f4ac
# ╟─79674636-617c-11eb-0213-fb99e78e9f1d
# ╠═e5fb5a90-6ea0-11eb-24ee-19d13bd0e193
# ╠═e5fc9310-6ea0-11eb-3ae5-d3704e3018ad
# ╠═e5fa9740-6ea0-11eb-23c9-832717e87c63
# ╠═e5f9ace0-6ea0-11eb-010f-91331637006c
# ╟─8c3f1d38-617c-11eb-2820-f32c96e276c6
# ╠═70a023e0-6ef8-11eb-16ec-8fa3011d3242
# ╠═7605da32-6ef5-11eb-13cb-ad7270aa4995
# ╠═9acfda70-6ef8-11eb-0800-d1f21da4c85c
# ╠═eb6abaf0-6ef7-11eb-0cf8-bb56211984c5
# ╠═e643f450-6ef8-11eb-3858-fb9cf3e78cc6
# ╠═0fe93dd0-6efc-11eb-1366-85e28ccc33fa
# ╠═103e3ba0-6efc-11eb-18f0-bd3fb799bf3f
# ╠═10bf2b70-6efc-11eb-3c36-cb41af9baffb
