### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 86371c3e-6112-11eb-1660-f32994a6b1a5
using Distributions: pdf, Normal, Chi, logpdf

# ╔═╡ 3d005a40-6cf6-11eb-3f93-790090bbc880
using Plots: histogram, histogram!, plot!, savefig, plot

# ╔═╡ f7b41650-6d83-11eb-1a77-47a6aa9079a8
using Statistics: mean, var

# ╔═╡ dc2cf2d0-6dba-11eb-0207-f91c4c8dbd28
using LinearAlgebra: det, inv, I

# ╔═╡ aa84f014-6111-11eb-25e5-c7bec21824e9
md"""
# High-Dimensional Gaussians [20 pts]

In this question we will investigate how our intuition for samples from a Gaussian may break down in higher dimensions. Consider samples from a $D$-dimensional unit Gaussian

  $x \sim \mathcal{N}(0_D, I_D)$
where~$0_D$ indicates a column vector of~$D$ zeros and~$I_D$
is a ${D\times D}$ identity matrix.
"""


# ╔═╡ 0634fd58-6112-11eb-37f6-45112ee734ae
md"""
### Distance of Gaussian samples from origin

Starting with the definition of Euclidean norm, quickly show that the distance of $x$ from the origin is $\sqrt{ x ^ \intercal x }$
"""

# ╔═╡ 08b6ed2a-6112-11eb-3277-69f7c404be51
md"""
### Distribution of distances of Gaussian samples from origin

In low-dimensions our intuition tells us that samples from the unit Gaussian will be near the origin. 

1. Draw 10000 samples from a $D=1$ Gaussian
2. Compute the distance of those samples from the origin.
3. Plot a normalized histogram for the distance of those samples from the origin. 

Does this confirm your intuition that the samples will be near the origin?

Answer: Yes. By properties of Gaussian distribution, we know that the 99% of the sample points will have a distance less than 3 (3 stddevs). 
"""

# ╔═╡ 00ec92e0-6cfa-11eb-0375-537b94006094
begin
	p1 = plot(fmt=:png)
	x = randn(10000,1)	
	dist_single = sqrt.(x .* x)
	histogram!(dist_single, normalize=true, label="distance to origin")
	p1
end

# ╔═╡ 117c783a-6112-11eb-0cfc-bb24a3234baf
md"""
Draw 10000 samples from $D=\{1,2,3,10,100\}$ Gaussians and, on a single plot, show the normalized histograms for the distance of those samples from the origin. As the dimensionality of the Gaussian increases, what can you say about the expected distance of the samples from the Gaussian's mean (in this case, origin).

Answer: The expected distance increases, and is proportional to the square root of the Gaussian dimensionality.
"""

# ╔═╡ 149abbb2-6cfe-11eb-0ec3-1ddfcf39b161
begin
	p2 = plot(fmt=:png)
	for i in [1 2 3 10 100]
		y = randn(10000,i)	
		dist_multi = sqrt.(sum(y .* y, dims=2))
		histogram!(dist_multi, normalize=true, label="$i-dim")
	end
	p2
end

# ╔═╡ 387dc1de-6174-11eb-069c-e70e4483ea67
md"""
### Plot the $\chi$-distribution

From Wikipedia, if $x_i$ are $k$ independent, normally distributed random variables
with means $\mu_i$ and standard deviations $\sigma_i$ then the statistic $Y =
\sqrt{\sum_{i=1}^k(\frac{x_i-\mu_i}{\sigma_i})^2}$ is distributed according to the
[$\chi$-distribution](https://en.wikipedia.org/wiki/Chi_distribution)] 

On the previous normalized histogram, plot the probability density function (pdf) 
of the $\chi$-distribution for $k=\{1,2,3,10,100\}$.
"""

# ╔═╡ 67cc8c54-6174-11eb-02d1-95d31e908329
function norm_to_chi(x; mean=0., variance=1.)
	temp = (x .- mean)./variance
	return sqrt(sum(temp' * temp))
end

# ╔═╡ bde997f0-6d61-11eb-016a-6d6654b48c36
# Implementation of chi-distribution pdf
function chi_pdf(x; k=1)
	k2 = k
	if k % 2 == 1
		gamma = sqrt(pi)
	else
		gamma = 1
	end
	while k2/2 > 1
		gamma *= (k2/2 - 1)
		k2 -= 1
	end
	return exp(-x^2/2 + (k - 1) * log(x))/2^(k/2 - 1)/gamma
end

# ╔═╡ 10732810-6d6c-11eb-03ba-f94c873d451f
# Check implementation against standard library, unstable for larger values
# Chi will be used for all calculations afterwards 
begin
	new_dist = Chi(100)
	isapprox(pdf.(Chi(100), 0:0.01:100), chi_pdf.(0:0.01:100; k=100))
end

# ╔═╡ b6647650-6d5f-11eb-2421-1bf0ad88cdaf
begin
	for k in [1 2 3 10 100]
		plot!(0:0.01:20, pdf.(Chi(k), 0:0.01:20), label="$k -pdf", lw=2)
	end
	p2
end

# ╔═╡ 6865c7f2-6174-11eb-2d0b-c73f00d5c347
md"""
### Distribution of distance between samples

Taking two samples from the $D$-dimensional unit Gaussian,
$x_a, x_b \sim \mathcal{N}(  0_D, I_D)$ how is $x_a -  x_b$ distributed?
Using the above result about $\chi$-distribution, derive how $\vert \vert x _a -  x _b\vert \vert_2$ is distributed.


(Hint: start with a $\mathcal{X}$-distributed random variable and use the [change of variables formula](https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables).) 

Answer: $x_a - x_b \sim \mathcal{N}( 0_D, 2I_D)$ by properties of Gaussian. Then we know the $\chi$-distribution $X = \sqrt{\sum_{i=1}^k(\frac{x_{ai} - x_{bi}}{\sqrt{2}})^2}$ = $\frac{\vert \vert x _a - x_b \vert \vert_2}{\sqrt{2}}$

Want to find $Y$ = $\sqrt{2} X  = \vert \vert x _a - x_b \vert \vert_2$

By change of variable we get $f_Y(y)$ = $f_X(g(y)) g'(y)$ where $g(y) = \frac{y}{\sqrt{2}}$ and $g'(y) = \frac{1}{\sqrt{2}}$

There we get $f_Y(y) = \frac{1}{\sqrt{2}} f_X (\frac{y}{\sqrt{2}})$ where $f_X (x) \sim \chi$-distribution of degree D

The function $dist\_chi\_pdf$ below implements the following transformation
"""

# ╔═╡ 2e54fd70-6d82-11eb-2cb9-f36bd98c47b9
function dist_chi_pdf(y; k=1)
	return 1/sqrt(2) * pdf(Chi(k), y/sqrt(2))
end

# ╔═╡ 52eb69c0-6db5-11eb-3edd-a57fe4395e6c
function euclidean_norm(x, y)
	return sqrt.(sum((x .- y).^2, dims=2))
end

# ╔═╡ dc4a3644-6174-11eb-3e97-0143edb860f8
md"""
### Plot pdfs of distribution distances between samples 

For for $D=\{1,2,3,10,100\}$. 
    How does the distance between samples from a Gaussian behave as dimensionality increases?
    Confirm this by drawing two sets of $1000$ samples from the $D$-dimensional unit Gaussian.
    On the plot of the $\chi$-distribution pdfs, plot the normalized histogram of the distance between samples from the first and second set.

Answer: Distance increases and the expected distance is proportional to the square root of the dimensionality. Expected distance between two sample points = $\sqrt{2} \cdot$ expected distance to the origin
"""

# ╔═╡ 171207b6-6175-11eb-2467-cdfb7e1fd324
begin
	p3 = plot(fmt=:png)
	for _dim in [1 2 3 10 100]
		
		# Define sample and plotting range
		sample_1 = randn(1000, _dim)
		sample_2 = randn(1000, _dim)
		dist_range = 0:0.01:max(2 * sqrt(2) * sqrt(_dim))
		
		# Pdf
		plot!(dist_range, dist_chi_pdf.(dist_range; k=_dim), label="$_dim-pdf", lw=2)
		
		# Histogram
		histogram!(euclidean_norm(sample_1, sample_2), normalize=true, label="$_dim-sample", bins=30)
	end
	p3
end

# ╔═╡ 18adf0b2-6175-11eb-1753-a7f33f0d7ca3
md"""
### Linear interpolation between samples

Given two samples from a gaussian $x_a,x_b \sim \mathcal{N}(  0 _D, I_D)$ the
    linear interpolation between them $x_\alpha$ is defined as a function of  $\alpha\in[0,1]$

$$\text{lin\_interp}(\alpha, x _a, x _b) = \alpha  x _a + (1-\alpha) x _b$$

For two sets of 1000 samples from the unit gaussian in $D$-dimensions, plot the average log-likelihood along the linear interpolations between the pairs of samples as a function of $\alpha$.

(i.e. for each pair of samples compute the log-likelihood along a linear space of interpolated points between them, $\mathcal{N}(x_\alpha|0,I)$ for $\alpha \in [0,1]$. Plot the average log-likelihood over all the interpolations.)


Do this for $D=\{1,2,3,10,100\}$, one plot per dimensionality.
Comment on the log-likelihood under the unit Gaussian of points along the linear interpolation.
Is a higher log-likelihood for the interpolated points necessarily better?
Given this, is it a good idea to linearly interpolate between samples from a high dimensional Gaussian?

Answer: The log-likelihood is a bell-shaped curve that reaches its maximum when $\alpha = 0.5$. It is not a good idea to interpolate in higher dimensions because the interpolated points are in general closer to the origin, resulting in a higher likelihood at midpoint.
"""

# ╔═╡ abdc2200-6db9-11eb-25be-01f6db44870c
# Likelihood functions and plotting tools
begin
	function log_likelihood(x; mu, sigma)
		_dim = size(sigma)[1]
		const_term = -_dim/2 * log(2 * pi) - 1/2 * log(det(sigma))
		exp_term = -1/2 * (x - mu)' * inv(sigma) * (x - mu)
		return const_term + exp_term
	end
	
	function lin_interp_likelihood(alpha; xa, xb, mu, sigma)
		pre_summation = log_likelihood.(eachrow(alpha .* xa + (1 - alpha) .* xb); mu=mu, sigma=sigma)
		return mean(pre_summation)
	end
	
	function plot_lin_interp(_dim=1, sample_size=1000)
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(sample_size, _dim)
		sample_2 = randn(sample_size, _dim)
		alpha_range = 0:0.01:1
		
		p = plot(fmt=:png)
		plot!(alpha_range, lin_interp_likelihood.(alpha_range; xa=sample_1, xb=sample_2, mu=m, sigma=cov), label="$_dim-dim_lin_interp")
		return p
	end
end

# ╔═╡ 033790a0-6e34-11eb-0e37-755b1c6c512e
plot_lin_interp(1, 1000)

# ╔═╡ 03e36100-6e34-11eb-2367-65e34c3e7866
plot_lin_interp(2, 1000)

# ╔═╡ 046c8e30-6e34-11eb-2a2b-73d76bb137cb
plot_lin_interp(3, 1000)

# ╔═╡ 04e03790-6e34-11eb-1167-7f44e71c2eb8
plot_lin_interp(10, 1000)

# ╔═╡ 0587c230-6e34-11eb-01e1-772e38ba7092
plot_lin_interp(100, 1000)

# ╔═╡ a738e7ba-6175-11eb-0103-fb6319b44ece
md"""
###  Polar Interpolation Between Samples

Instead we can interpolate in polar coordinates: For $\alpha\in[0,1]$ the polar interpolation is

$$\text{polar\_interp}(\alpha, x _a, x _b)=\sqrt{\alpha } x _a + \sqrt{(1-\alpha)} x _b$$

This interpolates between two points while maintaining Euclidean norm.


On the same plot from the previous question, plot the probability density of the polar interpolation between pairs of samples from two sets of 1000 samples from $D$-dimensional unit Gaussians for $D=\{1,2,3,10,100\}$. 

Comment on the log-likelihood under the unit Gaussian of points along the polar interpolation.
Give an intuitive explanation for why polar interpolation is more suitable than linear interpolation for high dimensional Gaussians. 
(For 6. and 7. you should have one plot for each $D$ with two curves on each).

Answer: Polar interpolation is more suitable because the log-likelihood of interpolated points are similar in magnitude with the start/end point. 
"""

# ╔═╡ d0b81a0c-6175-11eb-3005-811ab72f7077
begin 
	function polar_interp_likelihood(alpha; xa, xb, mu, sigma)
		pre_summation = log_likelihood.(eachrow(sqrt(alpha) .* xa + sqrt(1 - alpha) .* xb); mu=mu, sigma=sigma)
		return mean(pre_summation)
	end
	
	function plot_lin_polar_interp(_dim=1, sample_size=1000)
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(sample_size, _dim)
		sample_2 = randn(sample_size, _dim)
		alpha_range = 0:0.01:1
		
		p = plot(fmt=:png)
		plot!(alpha_range, lin_interp_likelihood.(alpha_range; xa=sample_1, xb=sample_2, mu=m, sigma=cov), label="$_dim-dim_lin_interp")
		plot!(alpha_range, polar_interp_likelihood.(alpha_range; xa=sample_1, xb=sample_2, mu=m, sigma=cov), label="$_dim-dim_polar_interp")
		return p
	end
end

# ╔═╡ 945ab760-6e34-11eb-1faf-c55c16e05abd
plot_lin_polar_interp(1, 1000)

# ╔═╡ 9e457940-6e34-11eb-069f-b974242477c2
plot_lin_polar_interp(2, 1000)

# ╔═╡ 9ebe04a0-6e34-11eb-2087-85b206ae781f
plot_lin_polar_interp(3, 1000)

# ╔═╡ 9f714f10-6e34-11eb-0046-dfe17f553036
plot_lin_polar_interp(10, 1000)

# ╔═╡ 9fed35ce-6e34-11eb-34d2-eded9da6b873
plot_lin_polar_interp(100, 1000)

# ╔═╡ e3b3cd7c-6111-11eb-093e-7ffa8410b742
md"""

### Norm along interpolation

In the previous two questions we compute the average log-likelihood of the linear and polar interpolations under the unit gaussian.
Instead, consider the norm along the interpolation, $\sqrt{ x _\alpha^ \intercal x _\alpha}$.
As we saw previously, this is distributed according to the $\mathcal{X}$-distribution.
Compute and plot the average log-likelihood of the norm along the two interpolations under the the $\mathcal{X}$-distribution for $D=\{1,2,3,10,100\}$, 
i.e. $\mathcal{X}_D(\sqrt{ x _\alpha^ \intercal x _\alpha})$. 
There should be one plot for each $D$, each with two curves corresponding to log-likelihood of linear and polar interpolations.
How does the log-likelihood along the linear interpolation compare to the log-likelihood of the true samples (endpoints)?

Answer: 

For the lower dimensions the log-likelihood of the true samples is lower than of interpolations. For higher dimensions, the log-likelihood of true samples is much higher.
"""


# ╔═╡ 07176c60-6176-11eb-0336-db9f450ed67f
begin 
	function lin_norm_likelihood(alpha; xa, xb)

		_dim = size(xa)[2]

		lin_interp = alpha .* xa + (1 - alpha) .* xb		
		lin_interp_norm = sqrt.(sum(eachcol(lin_interp .* lin_interp)))
		return mean(logpdf.(Chi(_dim), lin_interp_norm))
	end
	
	function polar_norm_likelihood(alpha; xa, xb)
		_dim = size(xa)[2]
		polar_interp = sqrt(alpha) .* xa + sqrt(1 - alpha) .* xb
		
		polar_interp_norm = sqrt.(sum(eachcol(polar_interp .* polar_interp)))
		
		return mean(logpdf.(Chi(_dim), polar_interp_norm))
	end
		
	function plot_lin_polar_norm_interp(_dim=1, sample_size=1000)

		sample_1 = randn(sample_size, _dim)
		sample_2 = randn(sample_size, _dim)
		alpha_range = 0:0.01:1

		p = plot(fmt=:png)
		plot!(alpha_range, lin_norm_likelihood.(alpha_range; xa=sample_1, xb=sample_2) , label="$_dim-dim_lin_interp_norm")
		plot!(alpha_range, polar_norm_likelihood.(alpha_range; xa=sample_1, xb=sample_2), label="$_dim-dim_polar_interp_norm")
		return p
	end
end	

# ╔═╡ 0f66a7be-6e35-11eb-36de-156d435aceb7
plot_lin_polar_norm_interp(1, 1000)

# ╔═╡ 0fcb35f0-6e35-11eb-0e06-d94aef5503b5
plot_lin_polar_norm_interp(2, 1000)

# ╔═╡ 101dc2c0-6e35-11eb-14f5-ef408d3b43c5
plot_lin_polar_norm_interp(3, 1000)

# ╔═╡ 106d4250-6e35-11eb-2df7-0d3a4182fda6
plot_lin_polar_norm_interp(10, 1000)

# ╔═╡ 10d18260-6e35-11eb-15c6-45798d1ee53e
plot_lin_polar_norm_interp(100, 1000)

# ╔═╡ 14eeab30-6dc6-11eb-28d8-79ec84de8aa9
md"""
### Supplementary plots for finding answers to questions above
"""

# ╔═╡ 559df772-6e35-11eb-3809-a118b5e4a11c
begin
	p4 = plot(fmt=:png)
	for _dim in [1 2 3 10 100]
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(1000, _dim)
		sample_2 = randn(1000, _dim)
		alpha_range = 0:0.01:1
		plot!(alpha_range, lin_interp_likelihood.(alpha_range; xa=sample_1, xb=sample_2, mu=m, sigma=cov), label="$_dim-dim_lin")
	end
	p4
end

# ╔═╡ a0db6660-6e34-11eb-27c0-17c80e717cde
begin
	p5 = plot(fmt=:png)
	for _dim in [1 2 3 10 100]
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(1000, _dim)
		sample_2 = randn(1000, _dim)
		alpha_range = 0:0.01:1
		plot!(alpha_range, polar_interp_likelihood.(alpha_range; xa=sample_1, xb=sample_2, mu=m, sigma=cov), label="$_dim-dim_polar")
	end
	p5
end

# ╔═╡ 3cca1210-6e35-11eb-1cbf-138b03e3b4c3
begin
	p6 = plot(fmt=:png)
	for _dim in [1 2 3 10 100]
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(1000, _dim)
		sample_2 = randn(1000, _dim)
		alpha_range = 0:0.01:1
		plot!(alpha_range, lin_norm_likelihood.(alpha_range; xa=sample_1, xb=sample_2), label="$_dim-dim_lin_interp_norm")
	end
	p6
end

# ╔═╡ 5027f4d0-6e35-11eb-05b8-29fe2a44e992
begin
	p7 = plot(fmt=:png)
	for _dim in [1 2 3 10 100]
		m = repeat([0], _dim)
		cov = Matrix(I, _dim, _dim)
		sample_1 = randn(1000, _dim)
		sample_2 = randn(1000, _dim)
		alpha_range = 0:0.01:1
		plot!(alpha_range, polar_norm_likelihood.(alpha_range; xa=sample_1, xb=sample_2), label="$_dim-dim_polar_interp_norm")
	end
	p7
end

# ╔═╡ Cell order:
# ╟─aa84f014-6111-11eb-25e5-c7bec21824e9
# ╟─0634fd58-6112-11eb-37f6-45112ee734ae
# ╟─08b6ed2a-6112-11eb-3277-69f7c404be51
# ╠═86371c3e-6112-11eb-1660-f32994a6b1a5
# ╠═3d005a40-6cf6-11eb-3f93-790090bbc880
# ╠═f7b41650-6d83-11eb-1a77-47a6aa9079a8
# ╠═dc2cf2d0-6dba-11eb-0207-f91c4c8dbd28
# ╠═00ec92e0-6cfa-11eb-0375-537b94006094
# ╠═117c783a-6112-11eb-0cfc-bb24a3234baf
# ╠═149abbb2-6cfe-11eb-0ec3-1ddfcf39b161
# ╟─387dc1de-6174-11eb-069c-e70e4483ea67
# ╠═67cc8c54-6174-11eb-02d1-95d31e908329
# ╠═bde997f0-6d61-11eb-016a-6d6654b48c36
# ╠═10732810-6d6c-11eb-03ba-f94c873d451f
# ╠═b6647650-6d5f-11eb-2421-1bf0ad88cdaf
# ╟─6865c7f2-6174-11eb-2d0b-c73f00d5c347
# ╠═2e54fd70-6d82-11eb-2cb9-f36bd98c47b9
# ╠═52eb69c0-6db5-11eb-3edd-a57fe4395e6c
# ╟─dc4a3644-6174-11eb-3e97-0143edb860f8
# ╠═171207b6-6175-11eb-2467-cdfb7e1fd324
# ╟─18adf0b2-6175-11eb-1753-a7f33f0d7ca3
# ╠═abdc2200-6db9-11eb-25be-01f6db44870c
# ╠═033790a0-6e34-11eb-0e37-755b1c6c512e
# ╠═03e36100-6e34-11eb-2367-65e34c3e7866
# ╠═046c8e30-6e34-11eb-2a2b-73d76bb137cb
# ╠═04e03790-6e34-11eb-1167-7f44e71c2eb8
# ╠═0587c230-6e34-11eb-01e1-772e38ba7092
# ╟─a738e7ba-6175-11eb-0103-fb6319b44ece
# ╠═d0b81a0c-6175-11eb-3005-811ab72f7077
# ╠═945ab760-6e34-11eb-1faf-c55c16e05abd
# ╠═9e457940-6e34-11eb-069f-b974242477c2
# ╠═9ebe04a0-6e34-11eb-2087-85b206ae781f
# ╠═9f714f10-6e34-11eb-0046-dfe17f553036
# ╠═9fed35ce-6e34-11eb-34d2-eded9da6b873
# ╟─e3b3cd7c-6111-11eb-093e-7ffa8410b742
# ╠═07176c60-6176-11eb-0336-db9f450ed67f
# ╠═0f66a7be-6e35-11eb-36de-156d435aceb7
# ╠═0fcb35f0-6e35-11eb-0e06-d94aef5503b5
# ╠═101dc2c0-6e35-11eb-14f5-ef408d3b43c5
# ╠═106d4250-6e35-11eb-2df7-0d3a4182fda6
# ╠═10d18260-6e35-11eb-15c6-45798d1ee53e
# ╟─14eeab30-6dc6-11eb-28d8-79ec84de8aa9
# ╠═559df772-6e35-11eb-3809-a118b5e4a11c
# ╠═a0db6660-6e34-11eb-27c0-17c80e717cde
# ╠═3cca1210-6e35-11eb-1cbf-138b03e3b4c3
# ╠═5027f4d0-6e35-11eb-05b8-29fe2a44e992
