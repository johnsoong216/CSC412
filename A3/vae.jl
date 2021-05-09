### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 4475c019-0074-43ee-9a3d-1a664e8424d5
using Flux

# ╔═╡ a7a0e082-34ba-4b73-961f-f7612704e0db
using MLDatasets: MNIST

# ╔═╡ c70eaa72-90ad-11eb-3600-016807d53697
using StatsFuns: log1pexp #log(1 + exp(x))

# ╔═╡ 0a761dc4-90bb-11eb-1f6c-fba559ed5f66
using Plots: plot, plot!, scatter, heatmap, grid ##

# ╔═╡ 6b792931-933b-4722-a614-4b2285b0f841
using Random: shuffle

# ╔═╡ 9e368304-8c16-11eb-0417-c3792a4cd8ce
md"""
# Assignment 3: Variational Autoencoders

- Student Name: Qien Song
- Student #: 1003974728
- Collaborators:

## Background

In this assignment we will implement and investigate a Variational Autoencoder as introduced by Kingma and Welling in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).


### Data: Binarized MNIST

In this assignment we will consider an  MNIST dataset of $28\times 28$ pixel images where each pixel is **either on or off**.

The binary variable $x_i \in \{0,1\}$ indicates whether the $i$-th pixel is off or on.

Additionally, we also have a digit label $y \in \{0, \dots, 9\}$. Note that we will not use these labels for our generative model. We will, however, use them for our investigation to assist with visualization.

### Tools

In previous assignments you were required to implement a simple neural network and gradient descent manually. In this assignment you are permitted to use a machine learning library for convenience functions such as optimizers, neural network layers, initialization, dataloaders.

However, you **may not use any probabilistic modelling elements** implemented in these frameworks. You cannot use `Distributions.jl` or any similar software. In particular, sampling from and evaluating probability densities under distributions must be written explicitly by code written by you or provided in starter code.
"""

# ╔═╡ 54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# load the original greyscale digits
# train_digits = Flux.Data.MNIST.images(:train)

# ╔═╡ 3c8797df-374c-4873-85dc-c8f97a6a4e9f
begin
	# load the original greyscale digits due to incompatibility with Flux
	train_digits, train_labels = MNIST.traindata()
	
	# Transform the horizontally-displayed data to vertically-displayed
	train_digits = permutedims(train_digits, (2, 1, 3))
	train_digits = train_digits[end:-1:1, :, :]
end

# ╔═╡ cdc3609d-a7f1-40a4-8969-3a2808af1085
greyscale_MNIST = reshape(train_digits, (28 * 28, size(train_labels)[1]))

# ╔═╡ c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# binarize digits
binarized_MNIST = greyscale_MNIST .> 0.5

# ╔═╡ 9e7e46b0-8e84-11eb-1648-0f033e4e6068
# partition the data into batches of size BS
BS = 200

# ╔═╡ 743d473c-8c1f-11eb-396d-c92cacb0235b
# batch the data into minibatches of size BS
batches = Flux.Data.DataLoader(binarized_MNIST, batchsize=BS)

# ╔═╡ db655546-8e84-11eb-21df-25f7c8e82362
# confirm dimensions are as expected (D,BS)
size(first(batches))

# ╔═╡ 2093080c-8e85-11eb-1cdb-b35eb40e3949
md"""
## Model Definition

Each element in the data $x \in D$ is a vector of $784$ pixels. 
Each pixel $x_d$ is either on, $x_d = 1$ or off $x_d = 0$.

Each element corresponds to a handwritten digit $\{0, \dots, 9\}$.
Note that we do not observe these labels, we are *not* training a supervised classifier.

We will introduce a latent variable $z \in \mathbb{R}^2$ to represent the digit.
The dimensionality of this latent space is chosen so we can easily visualize the learned features. A larger dimensionality would allow a more powerful model. 


- **Prior**: The prior over a digit's latent representation is a multivariate standard normal distribution. $p(z) = \mathcal{N}(z \mid \mathbf{0}, \mathbf{1})$
- **Likelihood**: Given a latent representation $z$ we model the distribution over all 784 pixels as the product of independent Bernoulli distributions parametrized by the output of the "decoder" neural network $f_\theta(z)$.
```math
p_\theta(x \mid z) = \prod_{d=1}^{784} \text{Ber}(x_d \mid f_\theta(z)_d)
```

### Model Parameters

Learning the model will involve optimizing the parameters $\theta$ of the "decoder" neural network, $f_\theta$. 

You may also use library provided layers such as `Dense` [as described in the documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Stacking-It-Up-1). 

Note that, like many neural network libraries, Flux avoids explicitly providing parameters as function arguments, i.e. `neural_net(z)` instead of `neural_net(z, params)`.

You can access the model parameters `params(neural_net)` for taking gradients `gradient(()->loss(data), params(neural_net))` and updating the parameters with an [Optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/).

However, if all this is too fancy feel free to continue using your implementations of simple neural networks and gradient descent from previous assignments.

"""

# ╔═╡ 45bc7e00-90ac-11eb-2d62-092a13dd1360
md"""
### Numerical Stability

The Bernoulli distribution $\text{Ber}(x \mid \mu)$ where $\mu \in [0,1]$ is difficult to optimize for a few reasons.

We prefer unconstrained parameters for gradient optimization. This suggests we might want to transform our parameters into an unconstrained domain, e.g. by parameterizing the `log` parameter.

We also should consider the behaviour of the gradients with respect to our parameter, even under the transformation to unconstrained domain. For instance a poor transformation might encourage optimization into regions where gradient magnitude vanishes. This is often called "saturation".

For this reasons we should use a numerically stable transformation of the Bernoulli parameters. 
One solution is to parameterize the "logit-means": $y = \log(\frac{\mu}{1-\mu})$.

We can exploit further numerical stability, e.g. in computing $\log(1 + exp(x))$, using library provided functions `log1pexp`
"""


# ╔═╡ e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# Numerically stable bernoulli density, why do we do this?
function bernoulli_log_density(x, logit_means)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
	return x .* logit_means .- log1pexp.(logit_means)
end

# ╔═╡ 3b07d20a-8e88-11eb-1956-ddbaaf178cb3
md"""
## Model Implementation

- `log_prior` that computes the log-density of a latent representation under the prior distribution.
- `decoder` that takes a latent representation $z$ and produces a 784-dimensional vector $y$. This will be a simple neural network with the following architecture: a fully connected layer with 500 hidden units and `tanh` non-linearity, a fully connected output layer with 784-dimensions. The output will be unconstrained, no activation function.
- `log_likelihood` that given an array binary pixels $x$ and the output from the decoder, $y$ corresponding to "logit-means" of the pixel Bernoullis $y = log(\frac{\mu}{1-\mu})$ compute the **log-**likelihood under our model. 
- `joint_log_density` that uses the `log_prior` and `log_likelihood` and gives the log-density of their joint distribution under our model $\log p_\theta(x,z)$.

Note that these functions should accept a batch of digits and representations, an array with elements concatenated along the last dimension.
"""

# ╔═╡ 83b83990-bfc6-4563-a8b7-38c955c4c28f
function gaussian_log_likelihood(μ, σ, x)
"""
compute log-likelihood of x under N(μ,σ)
"""
	constant_term = -1/2 * (log(2*pi) .+ 2 .* log.(σ))
	exp_term = -1 ./ (2 .* σ.^2) .* (x .- μ).^2
	return sum(constant_term .+ exp_term, dims=1)
end

# ╔═╡ 83a36c18-8ca8-4c9e-a7b2-6b7018e68694
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# ╔═╡ 32246440-2f30-47a1-a5dc-9a989ad726e3
function log_prior(zs; μ=0, logσ=0)
"""
calculate prior log-likelihood given mean and log_sigma
"""
	return gaussian_log_likelihood(μ, exp.(logσ), zs)
end

# ╔═╡ 3b386e56-90ac-11eb-31c2-29ba365a6967
# Define Decoder dimension
Dz, Dh, Ddata = 2, 500, 28^2

# ╔═╡ d7415d20-90af-11eb-266b-b3ea86750c98
begin
	# Decoder network: 2 -> tanh -> 500 -> identity -> 784
	layer1 = Dense(Dz, Dh, tanh)
	layer2 = Dense(Dh, Ddata, identity)
	decoder = Chain(layer1, layer2)
end 

# ╔═╡ 5b8721dc-8ea3-11eb-3ace-0b13c00ce256
function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	θ = decoder(z)
	return sum(bernoulli_log_density(x, θ), dims=1)
end

# ╔═╡ 76699abf-68c2-47fe-94f3-609e1dfec2c0
function joint_log_density(x, z)
	return log_likelihood(x, z) + log_prior(z)
end

# ╔═╡ b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
md"""
## Amortized Approximate Inference with Learned Variational Distribution

Now that we have set up a model, we would like to learn the model parameters $\theta$.
Notice that the only indication for *how* our model should represent digits in $z \in \mathbb{R}^2$ is that they should look like our prior $\mathcal{N}(0,1)$.

How should our model learn to represent digits by 2D latent codes? 
We want to maximize the likelihood of the data under our model $p_\theta(x) = \int p_\theta(x,z) dz = \int p_\theta(x \mid z)p(z) dz$.

We have learned a few techniques to approximate these integrals, such as sampling via MCMC. 
Also, 2D is a low enough latent dimension, we could numerically integrate, e.g. with a quadrature.

Instead, we will use variational inference and find an approximation $q_\phi(z) \approx p_\theta(z \mid x)$. This approximation will allow us to efficiently estimate our objective, the data likelihood under our model. Further, we will be able to use this estimate to update our model parameters via gradient optimization.

Following the motivating paper, we will define our variational distribution as $q_\phi$ also using a neural network. The variational parameters, $\phi$ are the weights and biases of this "encoder" network.

This encoder network $q_\phi$ will take an element of the data $x$ and give a variational distribution over latent representations. In our case we will assume this output variational distribution is a fully-factorized Gaussian.
So our network should output the $(\mu, \log \sigma)$.

To train our model parameters $\theta$ we will need also train variational parameters $\phi$.
We can do both of these optimization tasks at once, propagating gradients of the loss to update both sets of parameters.

The loss, in this case, no longer being the data likelihood, but the Evidence Lower BOund (ELBO).

1. Implement `log_q` that accepts a representation $z$ and parameters $\mu, \log \sigma$ and computes the logdensity under our variational family of fully factorized guassians.
1. Implement `encoder` that accepts input in data domain $x$ and outputs parameters to a fully-factorized guassian $\mu, \log \sigma$. This will be a neural network with fully-connected architecture, a single hidden layer with 500 units and `tanh` nonlinearity and fully-connected output layer to the parameter space.
2. Implement `elbo` which computes an unbiased estimate of the Evidence Lower BOund (using simple monte carlo and the variational distribution). This function should take the model $p_\theta$, the variational model $q_\phi$, and a batch of inputs $x$ and return a single scalar averaging the ELBO estimates over the entire batch.
4. Implement simple loss function `loss` that we can use to optimize the parameters $\theta$ and $\phi$ with `gradient`. We want to maximize the lower bound, with gradient descent. (This is already implemented)

"""

# ╔═╡ 81b83d2a-7293-4161-98b3-4063b5d4c1d9
function log_q(z, q_μ, q_logσ)
	return gaussian_log_likelihood(q_μ, exp.(q_logσ), z)
end

# ╔═╡ a41bcdd5-fb0c-4b1e-8ad5-f97bc10a267f
function unpack_params(encoded_vals)
	"""unpack the encoded values"""
	q_μ = encoded_vals[1:2, :]
	q_logσ = encoded_vals[3:4, :]
return q_μ, q_logσ
end

# ╔═╡ c1585da4-c442-422a-8afc-43422845e87c
begin
	# Decoder network: 784 -> tanh -> 500 -> identity -> 4 -> 2 x 2
	layer3 = Dense(Ddata, Dh, tanh)
	layer4 = Dense(Dh, Dz * 2, identity)
	encoder = Chain(layer3, layer4, unpack_params)
end 

# ╔═╡ ccf226b8-90b6-11eb-15a2-d30c9e27aebb
function elbo(x)
	# Batch Size
    num_samples = size(x)[2]
	
	# Variational parameters from data
	q_μ, q_logσ = encoder(x) 
	
	# Sample from variational distribution
    z = randn(2, num_samples) .* exp.(q_logσ) .+ q_μ
	
	# Joint likelihood of z and x under model
    joint_ll = joint_log_density(x, z)
	
	# likelihood of z under variational distribution
    log_q_z = log_q(z, q_μ, q_logσ)
	
	# Scalar value, mean variational evidence lower bound over batch
    elbo_estimate = sum(joint_ll - log_q_z)/num_samples 
  return elbo_estimate
end

# ╔═╡ f00b5444-90b6-11eb-1e0d-5d034735ec0e
function loss(x)
  return -elbo(x)
end

# ╔═╡ 70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
md"""
## Optimize the model and amortized variational parameters

If the above are implemented correctly, stable numerically, and differentiable automatically then we can train both the `encoder` and `decoder` networks with graident optimzation.

We can compute `gradient`s of our `loss` with respect to the `encoder` and `decoder` parameters `theta` and `phi`.

We can use a `Flux.Optimise` provided optimizer such as `ADAM` or our own implementation of gradient descent to `update!` the model and variational parameters.

Use the training data to learn the model and variational networks.
"""

# ╔═╡ 5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
function train!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	
	for epoch in 1:nepochs
		total_loss = 0
		num_batches = 0
		for batch in data
			# compute gradient wrt loss
			grad = Flux.gradient(params) do
			batch_loss = loss(batch)
			end
			# update parameters  
			Flux.update!(opt, params, grad)
			
			# Keep track of losses and batches
			batch_loss = loss(batch)
			total_loss += batch_loss
			num_batches += 1
		end
	
	# Avg Log loss over each batch
	epoch_loss = total_loss/num_batches
	@info "Epoch $epoch: epoch_loss:$epoch_loss" 

	# Optional: log loss using @info "Epoch $epoch: loss:..."
	# Optional: visualize training progress with plot of loss
	# Optional: save trained parameters to avoid retraining later
	end
	# return nothing, this mutates the parameters of enc and dec!
end

# ╔═╡ c86a877c-90b9-11eb-31d8-bbcb71e4fa66
train!(encoder, decoder, batches, nepochs=5)

# ╔═╡ 17c5ddda-90ba-11eb-1fce-93b8306264fb
md"""
## Visualizing the Model Learned Representation

We will use the model and variational networks to visualize the latent representations of our data learned by the model.

We will use a variatety of qualitative techniques to get a sense for our model by generating distributions over our data, sampling from them, and interpolating in the latent space.
"""

# ╔═╡ 1201bfee-90bb-11eb-23e5-af9a61f64679
md"""
### 1. Latent Distribution of Batch

1. Use `encoder` to produce a batch of latent parameters $\mu, \log \sigma$
2. Take the 2D mean vector $\mu$ for each latent parameter in the batch.
3. Plot these mene vectors in the 2D latent space with a scatterplot
4. Colour each point according to its "digit class label" 0 to 9.
5. Display a single colourful scatterplot
"""

# ╔═╡ d908c2f4-90bb-11eb-11b1-b340f58a1584
begin
	
	# Get variational parameter
	q_μ, q_logσ = encoder(binarized_MNIST[:, 1:5000])
	
	# Plot the 2D mean vector
	scatter(reshape(q_μ[2, :], :, 1), 
		reshape(q_μ[1, :], :, 1),
		xlabel = "Latent dimension μ1",
		ylabel = "Latent dimension μ2",
	    group = train_labels[1:5000],
	    title="Distribution of Latent Representation of Mean Vector First 5K labels"
		)
end

# ╔═╡ dcedbba4-90bb-11eb-2652-bf6448095107
md"""
### 2. Visualizing Generative Model of Data

1. Sample 10 $z$ from the prior $p(z)$.
2. Use the model to decode each $z$ to the distribution logit-means over $x$.
3. Transform the logit-means to the Bernoulli means $\mu$. (this does not need to be efficient)
4. For each $z$, visualize the $\mu$ as a $28 \times 28$ greyscale images.
5. For each $z$, sample 3 examples from the Bernoulli likelihood $x \sim \text{Bern}(x \mid \mu(z))$.
6. Display all plots in a single 10 x 4 grid. Each row corresponding to a sample $z$. Do not include any axis labels.
"""

# ╔═╡ bf61cba5-60eb-4d8e-800e-03bbc25960e2
function plot_greyscale(plot_arr; size=(660, 280))
	"""returns a greyscale image"""
	return heatmap(plot_arr, color=:grays, aspect_ratio=1, grid=false, ticks=false, axis=false, legend=false, size=size)
end

# ╔═╡ fd806d0c-f81a-414b-8622-edb03ac8f811
let
	
	# Define the number of samples from prior and bernoulli
	num_prior_samples = 10
	num_bernoulli_samples = 3
	
	# Sample 10z from prior distribution
	sample_z = randn(2, num_prior_samples)
	
	# Decode each z to obtain logit_means
	logit_means = decoder(sample_z)
	
	# Convert logit_mean to bernoulli_mean
	bernoulli_means = 1 ./ (1 ./ exp.(logit_means) .+ 1)
	
	# Initialize an array to concatenate image arrays vertically
	verti_arr = reshape([], 0, 28 * num_prior_samples)
	
	
	for i in 1:num_bernoulli_samples + 1
		
		# Initialize an array to concatenate image arrays horizontally
		horiz_arr = reshape([], 28, 0 )  
		for j in 1:num_prior_samples
			
			# Generate the bernoulli_mean 
			if i == num_bernoulli_samples + 1
				img_arr = bernoulli_means[:, j]
			
			# Sample from the bernoulli_mean
			else
				img_arr = rand(784) .< bernoulli_means[:, j]
			end
			
			img_arr = reshape(img_arr, 28, 28)
			horiz_arr = cat(horiz_arr, img_arr, dims=2)

		end
		
		verti_arr = cat(verti_arr, horiz_arr, dims=1)
	end
	
	# Display the 10 x 4 grid
	plot_greyscale(verti_arr)
end

# ╔═╡ 82b0a368-90be-11eb-0ddb-310f332a83f0
md"""
### 3. Visualizing Regenerative Model and Reconstruction

1. Sample 4 digits from the data $x \sim \mathcal{D}$
2. Encode each digit to a latent distribution $q_\phi(z)$
3. For each latent distribution, sample 2 representations $z \sim q_\phi$
4. Decode each $z$ and transform to the Bernoulli means $\mu$
5. For each $\mu$, sample 1 "reconstruction" $\hat x \sim \text{Bern}(x \mid \mu)$
6. For each digit $x$ display (28x28) greyscale images of $x, \mu, \hat x$
"""

# ╔═╡ 1f74b033-ab22-4c9d-97e9-17bbe129bbb2
let
	
	# Sample 4 digits from the data
	new_sample = binarized_MNIST[:, shuffle(1:size(binarized_MNIST)[2])][:, 1:4]
	# Encode each digit to obtain variational parameters
	new_μ, new_logσ = encoder(new_sample)
	
	# Sample 2 representations z1, z2 from the variational parameters
	z1 = randn(2, 4) .* exp.(new_logσ) .+ new_μ
	z2 = randn(2, 4) .* exp.(new_logσ) .+ new_μ
	
	# Find logit mean of each representation
	logit_means_1 = decoder(z1)
	logit_means_2 = decoder(z2)
	
	# Find bernoulli mean of each representation
	bernoulli_means_1 = 1 ./ (1 ./ exp.(logit_means_1) .+ 1)
	bernoulli_means_2 = 1 ./ (1 ./ exp.(logit_means_2) .+ 1)	
	
	# Initialize an array to concatenate image arrays vertically
	verti_arr = reshape([], 0, 28 * 5)
	
	for i in 1:4
		
		# Initialize an array to concatenate image arrays horizontally
		horiz_arr = reshape([], 28, 0 )  
		for j in 1:5
			
			# Plot actual sample
			if j == 1
				img_arr = new_sample[:, i]
			end
			
			# Plot the bernoulli mean of the 1st representation
			if j == 2
				img_arr = bernoulli_means_1[:, i]
			end
			
			# Plot a sample from the bernoulli mean of the 1st representation
			if j == 3
				img_arr = rand(784) .< bernoulli_means_1[:, i]
			end
			
			# Plot the bernoulli mean of the 2nd representation
			if j == 4
				img_arr = bernoulli_means_2[:, i]
			end
			
			# Plot a sample from the bernoulli mean of the 2nd representation
			if j == 5
				img_arr = rand(784) .< bernoulli_means_2[:, i]
			end
			
			
			img_arr = reshape(img_arr, 28, 28)
			horiz_arr = cat(horiz_arr, img_arr, dims=2)
		end
		verti_arr = cat(verti_arr, horiz_arr, dims=1)
	end
	
	# Display the 5 x 4 grid
	plot_greyscale(verti_arr)
end

# ╔═╡ 02181adc-90c1-11eb-29d7-736dce72a0ac
md"""
### 4. Latent Interpolation Along Lattice

1. Produce a $50 \times 50$ "lattice" or collection of cartesian coordinates $z = (z_x, z_y) \in \mathbb{R}^2$.
2. For each $z$, decode and transform to a 28x28 greyscale image of the Bernoulli means $\mu$
3. Each point in the `50x50` latent lattice corresponds now to a `28x28` greyscale image. Concatenate all these images appropriately.
4. Display a single `1400x1400` pixel greyscale image corresponding to the learned latent space.
"""

# ╔═╡ 3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
let 
	dim1 = 49
	dim2 = 49
	
	# Create the lattice by finding the Cartesian product of (z1, z2), both incrementing from -3 to 3 with a predetermined interval
	latent_dim_1 = -2.5:(5/dim1):2.5
	latent_dim_2 = -2.5:(5/dim2):2.5
	z = [repeat(latent_dim_1, inner=[size(latent_dim_2,1)]) repeat(latent_dim_2, outer=[size(latent_dim_1,1)])]
	z = reshape(z, (dim1 + 1, dim2 + 1, 2))
	
	# Initialize an array to concatenate image arrays vertically	
	verti_arr = reshape([], 0, 28 * (dim1 + 1))
	
	for i in 1:dim1 + 1
		
		# Initialize an array to concatenate image arrays horizontally
		horiz_arr = reshape([], 28, 0)  
		for j in 1:dim2 + 1
			
			# Find the logit mean and convert to bernoulli mean
			logit_mean = decoder(z[i, j, :])
			img_arr = 1 ./ (1 ./ exp.(logit_mean) .+ 1)
			
			img_arr = reshape(img_arr, 28, 28)
			horiz_arr = cat(horiz_arr, img_arr, dims=2)
		end
		verti_arr = cat(verti_arr, horiz_arr, dims=1)
	end
	
	# Plot the 50 x 50 lattice
	plot_greyscale(verti_arr, size=(660, 660))
end 

# ╔═╡ Cell order:
# ╟─9e368304-8c16-11eb-0417-c3792a4cd8ce
# ╠═4475c019-0074-43ee-9a3d-1a664e8424d5
# ╠═a7a0e082-34ba-4b73-961f-f7612704e0db
# ╠═54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# ╠═3c8797df-374c-4873-85dc-c8f97a6a4e9f
# ╠═cdc3609d-a7f1-40a4-8969-3a2808af1085
# ╠═c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# ╠═9e7e46b0-8e84-11eb-1648-0f033e4e6068
# ╠═743d473c-8c1f-11eb-396d-c92cacb0235b
# ╠═db655546-8e84-11eb-21df-25f7c8e82362
# ╟─2093080c-8e85-11eb-1cdb-b35eb40e3949
# ╟─45bc7e00-90ac-11eb-2d62-092a13dd1360
# ╠═c70eaa72-90ad-11eb-3600-016807d53697
# ╠═e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# ╟─3b07d20a-8e88-11eb-1956-ddbaaf178cb3
# ╠═83b83990-bfc6-4563-a8b7-38c955c4c28f
# ╠═83a36c18-8ca8-4c9e-a7b2-6b7018e68694
# ╠═32246440-2f30-47a1-a5dc-9a989ad726e3
# ╠═3b386e56-90ac-11eb-31c2-29ba365a6967
# ╠═d7415d20-90af-11eb-266b-b3ea86750c98
# ╠═5b8721dc-8ea3-11eb-3ace-0b13c00ce256
# ╠═76699abf-68c2-47fe-94f3-609e1dfec2c0
# ╟─b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
# ╠═81b83d2a-7293-4161-98b3-4063b5d4c1d9
# ╠═a41bcdd5-fb0c-4b1e-8ad5-f97bc10a267f
# ╠═c1585da4-c442-422a-8afc-43422845e87c
# ╠═ccf226b8-90b6-11eb-15a2-d30c9e27aebb
# ╠═f00b5444-90b6-11eb-1e0d-5d034735ec0e
# ╟─70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
# ╠═5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
# ╠═c86a877c-90b9-11eb-31d8-bbcb71e4fa66
# ╟─17c5ddda-90ba-11eb-1fce-93b8306264fb
# ╠═0a761dc4-90bb-11eb-1f6c-fba559ed5f66
# ╟─1201bfee-90bb-11eb-23e5-af9a61f64679
# ╠═d908c2f4-90bb-11eb-11b1-b340f58a1584
# ╟─dcedbba4-90bb-11eb-2652-bf6448095107
# ╠═bf61cba5-60eb-4d8e-800e-03bbc25960e2
# ╠═fd806d0c-f81a-414b-8622-edb03ac8f811
# ╟─82b0a368-90be-11eb-0ddb-310f332a83f0
# ╠═6b792931-933b-4722-a614-4b2285b0f841
# ╠═1f74b033-ab22-4c9d-97e9-17bbe129bbb2
# ╟─02181adc-90c1-11eb-29d7-736dce72a0ac
# ╠═3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
