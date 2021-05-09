### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 17feea80-8354-11eb-2662-094b31e3fe15
using StatsFuns: log1pexp

# ╔═╡ 4b5267e0-8a84-11eb-3107-c755af4d0e4d
using Distributions: pdf, MvNormal, logpdf, cdf, Normal

# ╔═╡ 388c4d90-8ccf-11eb-1de4-cff652810053
using Plots

# ╔═╡ 6497c1a0-8cd2-11eb-192a-4983db47d233
using Statistics: mean

# ╔═╡ 6625b4a0-8cd2-11eb-010b-9db4b0c4d172
using Zygote: gradient

# ╔═╡ 8bf55322-8cd7-11eb-17e8-4b4428fe1c3e
using Random: shuffle

# ╔═╡ 615c1994-836f-11eb-3d3e-af2f4b7b7d80
using CSV, DataFrames

# ╔═╡ 6c6b0aa6-82b2-11eb-3114-25412fb07e27
md"""
# Assignment 2: Variational Inference in the TrueSkill model

- Name: Qien Song
- StudentNumber: 1003974728  
- Collaborators: Ben Prystawski

## Goal

The goal of this assignment is to become familiar with the basics of Bayesian inference in large models with continuous latent variables, and implement stochastic variational inference with Automatic Differentiation.

## Outline

1. Introduce our problem and describe our model.
2. Implement our model.
3. Investigate the model on easy-to-visualize toy data.
4. Implement Stochastic Variational Inference (SVI) objective (ELBO) manually, using Automatic Differentiation (e.g. [Zygote.jl](https://github.com/FluxML/Zygote.jl)).
6. Use our SVI implementation to learn an approximate distribution inferred from the toy data and visualize.
7. Use SVI to perform approximate inference on real data. A collection of games played by by Woman Grandmasters on chess.com. The data was modified from [this Kaggle Dataset](https://www.kaggle.com/rohanrao/chess-games-of-woman-grandmasters).
8. Use variational approximation to estimate inference questions using the model.


## Background

We will implement a variant of the TrueSkill model, a player ranking system for competitive games originally developed for Halo 2.
It is a generalization of the Elo rating system in Chess.

Here are some readings you can familiarize yourself with the problem and their model. Note that we will use different inference methods than those discussed in these works (message passing):

- [The 2006 technical report at Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/TR-2006-80.pdf)
- [The 2007 NeurIPS conference paper introducing TrueSkill](http://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system.pdf)
- [The 2018 followup work TrueSkill 2](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/trueskill2.pdf)



Our assignment is based on [one developed by Carl Rasmussen at Cambridge for his course on probabilistic machine learning](http://mlg.eng.cam.ac.uk/teaching/4f13/1920/cw/coursework2.pdf).
In their assignment they implement and utilize a Gibbs sampling method for approximate inference. 
We will not implement an MCMC sampling method, instead using SVI for our approximate inference.

"""

# ╔═╡ 225c993e-82b2-11eb-3322-31a522cc8594
md"""
## 1. Problem and Model Description

We will consider a simplified version of the TrueSkill model to model the skill of a individual players $i$ at a 2-player game: chess.

We assume that each player has an unknown skill $z_i \in \mathbb{R}$.
These are called latent variables to our model since we cannot observe the players' skill directly.

Given our list of players $I$, we have no way *a priori* (before we observe data) to infer the players' unknown skill.
If we were familiar with the roster of Woman Grandmaster (WGM) chess players, though, we could incorporate our *prior information* about the players to suggest plausible skills.
However, these initial priors can be biased, e.g. player fame, charisma, playstyle.

Instead, we will simply assume that all players' skill is distributed according to a standard Gaussian distribution.

We also assume that all players' skills are *a priori* independent. 

Because we cannot monitor players' skill directly, we must collect other evidence.
We observe the players' performance against other players in a series of games. 
We will use the observed game data to update our model of the players' skill.

Our model of the likelihood that player $i$ beats player $j$ in a game of chess depends on our model of their respective skills $z_i$, $z_j$.
We will use the following likelihood model:

```math
p(\text{game(i wins, j loses)} \mid z_i, z_j) = \sigma(z_i - z_j)
```
where
```math
\sigma(x) = \frac{1}{1+\exp(-x)} 
```

Note that $log(1+exp(y))$ can suffer significant numerical instability if not implemented carefully. You should use a function like `StatsFuns.log1pexp` or its `numpy` equivalent. 

We also assume that each game is independent from the others, given the players' skill.

### Data

The observations of game outcomes are collected in an array.
Each column contains a pair of player indices, $i, j$ for the chess game.
The first index, player $i$ is the winner, and so $j$ lost the game.

If we observe $M$ games played then the data is collected into an array with shape $2 \times M$.

Note that data retains its dimension even as we observe more players.
Additional players are indentified by their index.
If we wanted to consider multiplayer games, like Halo 2, we could increase the first dimension, from our 2-player chess game.
"""


# ╔═╡ ce219a64-8350-11eb-37aa-1d156db00ce3
md"""
## 2.  Implementing the Model

- log-prior of skills zs
- log-likelihood of observing game with player i beating player j
- log-likelihood of observing collection of games with players' skills zs
- log-likelihood of joint distribution observed games and skills zs

"""

# ╔═╡ b0113760-8b8d-11eb-27d1-f3d7edba2028
function gaussian_log_likelihood(μ, σ, x)
"""
compute log-likelihood of x under N(μ,σ)
"""
constant_term = -1/2 * (log(2*pi) .+ 2 .* log.(σ))
exp_term = -1 ./ (2 .* σ.^2) .* (x .- μ).^2
return sum(constant_term .+ exp_term, dims=1)
end

# ╔═╡ 237e7586-8351-11eb-0810-8b09c322cf0c
function log_prior(zs; μ=0, logσ=0)
"""
calculate prior log-likelihood given mean and log_sigma
"""
  return gaussian_log_likelihood(μ, exp.(logσ), zs)
end

# ╔═╡ a08b1960-8358-11eb-1d60-615634e45184
function logp_i_beats_j(zi,zj)
  return -log1pexp.(zj - zi)
end

# ╔═╡ cdc987cc-8358-11eb-118f-0f4d7ca2e040
function all_games_log_likelihood(zs,games)
"""
calculate log_likelihood given evidence
"""
  zs_a = zs[games[1, :], :]#TODO
  zs_b =  zs[games[2, :], :]#TODO
  likelihoods = logp_i_beats_j.(zs_a, zs_b)
  return sum(likelihoods, dims=1) #TODO
end

# ╔═╡ 1392a8c4-8359-11eb-1639-29392237258c
function joint_log_density(zs,games; μ=0, logσ=0)
"""
log(joint) = log(prior) + log(evidence)
"""
  return log_prior(zs; μ=μ, logσ=logσ) + all_games_log_likelihood(zs, games)#TODO
end

# ╔═╡ 37c3b10e-8352-11eb-1c03-eddedcc516e0
md""" 
## 3. Visualize the Model on Toy Data

To check our understanding of the data and our model we consider a simple scenario.

### Toy Data

Let's model the chess skills for *only* two players, $A$ and $B$.
Restricting to 2-dimensions allows us to visualize the *posterior* distribution of our model.
Unlike the high-dimensional distributions we would model with real data, we can evaluate the model on a grid of points, like numerical integration, to produce a plot of contours for the possibly complicated posterior.

We provide a function `two_player_toy_games` which produces toy chess data for two players.
e.g. `two_player_toy_games(5,3)` produces a dataset where player $A$ wins 5 games and player $B$ wins 3 games.

### 2D Posterior Visualization

You can use the function `skillcontour!` to perform this grid of evaluations over the 2D latent space $z_A \times z_B$.
This will plot isocontours, curves of equal density, for the posterior distribution.

As well, `plot_line_equal_skill!` is provided to simply indicate the region of latent space corresponding to the players having equal skill, $z_A = z_B$.

(the default settings for these plotting functions may need modification.)
"""

# ╔═╡ 55c9ac2e-8381-11eb-2ee7-e564b39a2325
two_player_toy_games(p1_wins, p2_wins) = cat(collect.([repeat([1,2]',p1_wins)', repeat([2,1]',p2_wins)'])..., dims=2)

# ╔═╡ 5681e80e-835c-11eb-3e50-e32d41cb1ddb
function skillcontour!(f; colour=nothing)
  n = 100
  x = range(-3,stop=3,length=n)
  y = range(-3,stop=3,length=n)
  z_grid = Iterators.product(x,y) # meshgrid for contour
  z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim
  z = f.(z_grid)
  z = getindex.(z,1)'
  max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* max_z
  if colour==nothing
  p1 = contour!(x, y, z, fill=false, levels=levels)
  else
  p1 = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(p1)
end

# ╔═╡ 5617427e-835c-11eb-1ae3-ed0e13b24fe3
function plot_line_equal_skill!()
  plot!(range(-3, 3, length=200), range(-3, 3, length=200), label="Equal Skill")
end

# ╔═╡ e7f18c6e-835b-11eb-33ee-7de9b4f5071a
md"""
Plot the following distributions for two players $A$ and $B$.
In each, also plot the line of equal skill $z_A = z_B$.

Note, before plotting convert from log-density to density.
Since there is only 1 distribution per plot, we are not comparing distributions, we do not need to worry about normalization of the contours.
1. Isocontours of the prior distribution over players' skills. 
2. Isocontours of the likelihood function.
3. Isocontours of the posterior over players' skills given the observaton: `player A beat player B in 1 game`.
4. Isocontours of the posterior over players' skills given the observation: `player A beat player B in 10 games`.
5. Isocontours of the posterior over players' skills given the observation: `20 games were played, player A beat player B in 10 games`.
"""

# ╔═╡ f5e2fb76-8380-11eb-1a4b-1737bf6688bc
#1 Prior Distribution over Player Skills
begin
	exp_log_prior(zs) = exp.(log_prior(zs))
	plot(title="Two Player Skill Prior Distribution")
	skillcontour!(exp_log_prior)
	plot_line_equal_skill!()
end

# ╔═╡ f5a9cdd6-8380-11eb-07e1-d37f72198e39
#2 
begin
	exp_likelihood(zs) = exp.(logp_i_beats_j.(zs[1, :], zs[2, :]))
	plot(title="Two Player Skill Joint Distribution")
	skillcontour!(exp_likelihood)
	plot_line_equal_skill!()
end

# ╔═╡ f56d321a-8380-11eb-3918-a7e25c3b654d
#3 
begin
	exp_one_win(zs) = exp.(joint_log_density(zs, two_player_toy_games(1, 0)))
	plot(title="Two Player Skill Joint Distribution with Game Record 1:0")
	skillcontour!(exp_one_win)
	plot_line_equal_skill!()
end

# ╔═╡ d96c5102-8aaa-11eb-0131-3797c12c0ecc
#4 
begin
	exp_ten_win(zs) = exp.(joint_log_density(zs, two_player_toy_games(10, 0)))
	plot(title="Two Player Skill Joint Distribution with Game Record 10:0")
	skillcontour!(exp_ten_win)
	plot_line_equal_skill!()
end

# ╔═╡ f4ec68ec-8380-11eb-2a55-e1ed7b634ae8
#5
begin
	exp_equal_win(zs) = exp.(joint_log_density(zs, two_player_toy_games(10, 10)))
	plot(title="Two Player Skill Joint Distribution with Game Record 10:10")
	skillcontour!(exp_equal_win)
	plot_line_equal_skill!()
end

# ╔═╡ 08eb9304-835e-11eb-3798-ad7ecf016eb3
md"""
## 4. Stochastic Variational Inference with Automatic Differentiation

A nice quality of our Bayesian approach is that we separate the model specification from the (approximate) method we use for inference.

In the original TrueSkill paper they described a message passing strategy for approximate inference.
In Carl Rasmussen's assignment the students implement Gibbs sampling, a kind of Markov  Chain Monte Carlo method.

We will use gradient-based stochastic variational inference (SVI), a recent technique that is extremely successful in our domain. 

We will use SVI to produce an approximate posterior distribution to the, possibly complicated, posterior distributions specified by our model.

1. Implement the function `elbo` which computes an unbiased estimate of the Evidence Lower BOund. The ELBO is proportional to a KL divergence between the model posterior $p(z \mid data)$ and the approximate variational distribution $q_\phi(z \mid data)$.
2. Implement an objective function to optimize the parameters of the variational distribution. We will minimize a "loss" with gradient descent, the negative ELBO esitmated with $100$ samples. 
3. Write an optimization procedure `learn_and_vis_toy_variational_approx` that takes initial variational parameters and observed data and performs a number of iterations of gradient optimization where each iteration: 
    1. compute the gradient of the loss with respect to the variational parameters using AD
    2. update the variational parameters taking a `lr`-scaled step in the direction of the descending gradient.
    3. report the loss with the new parameters (using @info or print statements)
    4. on one plot axis plot the target posterior contours in red and the variational approximation contours in blue. `display` the plot or save to a folder. 

"""

# ╔═╡ 3a7ce2aa-8382-11eb-2029-f97ac18d1f97
function elbo(params,logp,num_samples)
	
  μ = params[1]
  logσ = params[2]
  num_players = size(params[2])[1]
  
  # Linear Transformation to Generate Samples
  samples = randn(num_players, num_samples) .* exp.(logσ) .+ μ
  
  # joint distribution of games and skills
  logp_estimate = logp(samples) 

  # prior distribution q_phi(z)  
  logq_estimate = log_prior(samples; μ=μ, logσ=logσ)
  
  # Expectation
  return mean(logp_estimate - logq_estimate)
end

# ╔═╡ 09d97bd2-835e-11eb-2545-3b34186ea79d
# Conveinence function for taking gradients 
function neg_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q, 
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end

# ╔═╡ 4491de6a-8382-11eb-3c4f-f3edbbd8b6d9
function learn_and_vis_toy_variational_approx(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params 
  final_loss = 0

  anim = @animate for i in 1:num_itrs
	
	# Calculate cur_elbo
	cur_elbo = neg_elbo(params_cur; 
					games=toy_evidence, 
					num_samples=num_q_samples)
	
	# Gradients of variational objective with respect to parameters
    grad_params = gradient(params -> neg_elbo(params; games=toy_evidence, num_samples=num_q_samples), params_cur)[1]
	
	# Update paramters with lr-sized step in descending gradient
    params_cur =  params_cur .- lr .* grad_params
	
	# Report the current elbo during training
	@info "elbo: $cur_elbo"
	
	# Define Target Posterior and Variational Inference
	target_posterior(zs) = exp.(joint_log_density(zs, toy_evidence))
	var_approx(zs) = exp.(log_prior(zs; μ=params_cur[1],logσ=params_cur[2]))
    
	# Plot true posterior in red and variational in blue
    plot();
	skillcontour!(target_posterior, colour=:red)
	skillcontour!(var_approx, colour=:blue)
	plot_line_equal_skill!()
 
	final_loss = cur_elbo
  end
  return (params_cur, final_loss),  anim
end

# ╔═╡ a5b747c8-8363-11eb-0dcf-2311ebef4a2a
md"""
## 5. Visualizing SVI on Two Player Toy

For the following collections of observed toy data of two player games learn an variational distribution to approximate the model posterior. 

Using the plots produced in the training SVI procedure, one plot per iteration, create an animation that shows the variational distribution converging to its final approximation. 
In your final report, please simply plot the the learned variational distribution in at the final iteration.

Each of these require its own set of initial variational parameters and variational optimization procedure. The plots should have contours for two distributions, model posterior (red) and variational approximation (blue). To allow useful visualization do **not** normalize the contours to the same scale.

1. Report the final loss and plot the posteriors for the oberservation: `player A beats player B in 1 game`
2. Report the final loss and plot the posteriors for the oberservation: `player A beats player B in 10 games`
3. Report the final loss and plot the posteriors for the oberservation: `player A wins 10 games player B wins 10 games`
"""

# ╔═╡ a4890ef8-8382-11eb-0fa3-df568c42ef1e
# Toy game
toy_mu = [-2., 3.] # Initial mu, can initialize randomly!

# ╔═╡ bc156e4a-8382-11eb-2fec-797f286ae96c
toy_ls = [0.5, 0.] # Initual log_sigma, can initialize randomly!

# ╔═╡ bc160bac-8382-11eb-11be-57f2f3484921
toy_params_init = (toy_mu, toy_ls)

# ╔═╡ c1f56960-8cb4-11eb-0990-8d76d5ccbb91


# ╔═╡ c1f495d0-8b92-11eb-385a-d59c7a75c1d7
function final_plot(params_final, toy_evidence, one_win_loss)
	"""
	Plot the Target Posterior and Variational Inference Distribution
	"""
	p = plot(title="Model Posterior Final Loss $one_win_loss");
	
	# Define Target Posterior and Variational Inference
	target_posterior(zs) = exp.(joint_log_density(zs, 			 			toy_evidence))
	var_approx(zs) = exp.(log_prior(zs;μ=params_final[1],logσ=params_final[2]))
	
	# Static Contour Plot
	skillcontour!(target_posterior, colour=:red)
	skillcontour!(var_approx, colour=:blue)
	plot_line_equal_skill!()
    return p
end

# ╔═╡ 54573350-8365-11eb-2069-ff2041af30c5
#1 Fit q with SVI observing player A winning 1 game
begin
	(one_win_params, one_win_loss), one_win_anim = learn_and_vis_toy_variational_approx(toy_params_init, two_player_toy_games(1, 0))
	gif(one_win_anim, "one_win_anim.gif", fps = 10)
end

# ╔═╡ c064ce10-8b92-11eb-04ea-cb2a41b785c5
begin
	final_plot(one_win_params, two_player_toy_games(1, 0), round(one_win_loss, digits=3))
end

# ╔═╡ 5588835a-8365-11eb-01bf-dba9289b3f59
#2 Fit q with SVI observing player A winning 10 games
begin
	(ten_win_params, ten_win_loss), ten_win_anim = learn_and_vis_toy_variational_approx(toy_params_init, two_player_toy_games(10, 0))
	gif(ten_win_anim, "ten_win_anim.gif", fps = 10)
end

# ╔═╡ 21523b30-8b94-11eb-3a3d-bf0f3b875cb4
begin
	final_plot(ten_win_params, two_player_toy_games(10, 0), round(ten_win_loss, digits=3))
end

# ╔═╡ 56d3d372-8365-11eb-21f8-b941b23056e1
#3 Fit q with SVI observing player A winning 10 games and player B winning 10 games
begin
	(equal_win_params, equal_win_loss), equal_win_anim = learn_and_vis_toy_variational_approx(toy_params_init, two_player_toy_games(10, 10))
	gif(equal_win_anim, "equal_win_anim.gif", fps = 10)
end

# ╔═╡ bef092b0-8b94-11eb-28ff-932b0d0a13c0
begin
	final_plot(equal_win_params, two_player_toy_games(10, 10), round(equal_win_loss, digits=3))
end

# ╔═╡ 61baa2fa-8365-11eb-1d51-cb9ba987cb00
md"""
## 6. Approximate Inference on Real Data

### Original Dataset

We will model the skills of players on chess.com from a collection of games played by Women GrandMasters since September 2009.

I have modified the dataset collected from the Kaggle Dataset: [Chess Games of Woman Grandmasters (2009 - 2021)](https://www.kaggle.com/rohanrao/chess-games-of-woman-grandmasters).

Note the extra information contained in the original dataset of game observations.
If we were familiar with the rules of chess when we designed our model, and aware of this extra data collected in our observations, we could describe a better model.
For example, 
- since White moves first, the White player has a considerable advantage
- not all games are observed as `win,lose`, some games end in draw due to repitition, or win by timeout.
- games have different timing rules.


Our model is simple and does not make use of that extra information in the original dataset.
For this assignment it will not be necessary to modify our model beyond the simple case. 
If you are interested in extending this, consider how you could adapt the model specification to incorporate these observations.

### Modified Dataset

My modifications to the data remove extra information about which player is White and simplifies the win conditions.

The data consists of two arrays:
- `games` is a $2 \times M$ collection of chess game outcomes, one game per column. The first row contains the indices of players who won, the second contains the indices of players who lost.
- `names` is a vector of player names, `names[i]` corresponds to the player chess.com username indicated by the values in `games`.

"""

# ╔═╡ 68d993cc-836f-11eb-2df1-c53cd5ea6ed4
games = collect(Array(CSV.read("games.csv", DataFrame))')

# ╔═╡ aa156070-8372-11eb-3d8c-4946a3e32d36
names = vec(Array(CSV.read("names.csv", DataFrame)))

# ╔═╡ 4e914c4a-8369-11eb-369e-af5416995cb8
md"""

### Variational Distribution

Use a fully-factorized Gaussian distribution for $q_\phi(z \mid data)$.

$$q_\phi(z \mid games) = \prod_i \mathcal{N}(z_i \mid \mu_i, \sigma_i)$$

**Note**, for numerical stability we work with the log_density. 
Also, for unconstrained optimization, we parameterize our Gaussian with $\log \sigma$, not variance or standard deviation.

You will need to implement this variational distribution with parameters $\phi = [\mu, \log \sigma]$ by defining `logq(z, params)` inside the `elbo`. 


Using the model and SVI method, we will condition our posterior on observations from the dataset.

In the previous Two Player Toy we were informed about the players' skill by observing games between them. Now we have many players playing games among eachother. For any two players, $i$ and $j$, answer yes or no to the following:

1. In general, is $p(z_i, z_j \mid \text{all games})$ proportional to $p(z_i, z_j, \text{all games})$?
2. In general, is $p(z_i, z_j \mid \text{all games})$ proportional to $p(z_i, z_j \mid \text{games between i and j})$? That is, do the games between player $i$ and $j$ provide all the information about the skills $z_i$ and $z_j$? 

Hint: consider the graphical model for three players, $i,j,k$ and condition on results of games between all 3. Examine the conditional independeinces in the model. (graphical model is not required for marks)

3. write a new optimization procedure `learn_variational_approx` like the one from the previous question, but **does not produce any plots**. As well, have the optimization procedure save the loss (negative ELBO) for each iteration into an array. Initialize a variational distribution and optimize it to learn the model joint distribution observing the chess games in the dataset. 

4. Plot the losses (negative ELBO estimate) obtained during variational optimization and report the loss of the final approximation.

5. We now have an approximate posterior over all our players' skill. Our variational family is simple, allowing us to easily compute an approximate mean and variance for all players' skill. Sort the players by mean skill and list the names of the 10 players with highest mean skill under our variational approximation. Use `sortperm`.

6. Plot the mean and variance of all players' skill, sorted by mean skill. There is no need to include the names of the players in the x-axis. Use `plot(means, yerror=exp.(logstds))` to add variance to the plot.

"""

# ╔═╡ f7513740-8cbb-11eb-1d33-a1d4f3f29b2b
function learn_variational_approx(init_params, chess_games; num_itrs=200, lr= 1e-2, num_q_samples=10, batch_size=10000)
  params_cur = init_params
  log_loss = []
  
  # Partition into minibatches
  batch_iterator(matrix, batchsize) = Iterators.partition(axes(matrix,2), batchsize)
  for i in 1:num_itrs
	
	cur_elbo = 0
	# Shuffle Data
	random_games = chess_games[:, shuffle(1:end)]
	
	# Iterate through minibatches
	for j in batch_iterator(random_games, batch_size)
		random_batch = random_games[:, j]
		cur_elbo += neg_elbo(params_cur; 
					games=random_batch, 
					num_samples=num_q_samples)
		# Gradients of variational objective wrt params
    	grad_params = gradient(params -> neg_elbo(params; games=random_batch, num_samples=num_q_samples), params_cur)[1]
		# Update parmaeters wite lr-sized steps in desending gradient direction
    	params_cur = params_cur .- lr .* grad_params
	end
		
	@info "Epoch: $i loss: $cur_elbo"
		
	# Save objective value with current parameters
    push!(log_loss, cur_elbo)
  end
  return params_cur, log_loss
end

# ╔═╡ 88de80c0-8cd8-11eb-2028-f1bf5fbff798
md"""
### Q1 Yes or No? 

Answer: Yes. According to Bayes Theorem, the posterior is proportional to the joint distribution, which equals evidence * likelihood. 
### Q2 Yes or No? 
Answer: No. The skills of player i and j are not solely dependent on the games they played against each other.
"""

# ╔═╡ 6af25166-8384-11eb-24d9-85aed2d14ee2
#3
# TODO: Initialize variational family  
begin
	init_mu = randn(size(names)[1]) * 0.01
	init_log_sigma = randn(size(names)[1]) * 0.01
	
	# random initialziation
	init_params = (init_mu, init_log_sigma) 
end

# ╔═╡ a34a1b64-8384-11eb-2e9f-0de11f1cfd4d
variational_params, log_loss = learn_variational_approx(init_params, games; num_itrs=100, lr=5e-4, batch_size=10000)

# ╔═╡ b5dc532a-8384-11eb-244e-556e2769179e
#4 Plot losses during ELBO optimization and report final loss
begin
	final_loss = round(log_loss[end], digits=3)
	plot(log_loss, title="Final Loss: $final_loss", label="log loss")
end

# ╔═╡ 40321e00-8c0a-11eb-2ecf-c30844d8dc7a


# ╔═╡ d19cc320-8c0c-11eb-3470-25dbd0bfbaeb
# Sort players by mean skill under our model and list top 10 players.
names[sortperm(variational_params[1], rev=true)[1:10]]

# ╔═╡ efce7f36-8384-11eb-2499-19ffdb51451f
#6 Plot mean and variance of all players, sorted by skill
begin
	means = sort(variational_params[1])
	logstds = variational_params[2][sortperm(variational_params[1])]
	plot(means, ribbon=exp.(logstds), label="Player Skill", title="Player Skill in Ascending Order")
end

# ╔═╡ f1e54852-836d-11eb-1847-13ab95054446
md"""
## 7. More Approximate Inference with our Model

Some facts from [The Queens of Chess Kaggle Notebook](https://www.kaggle.com/rohanrao/the-queens-of-chess) motivate a couple inference questions.


- `camillab` Camilla Baginskaite is a Lithuanian and American WGM and is the most active player in the dataset, over 60K games played in the last 12 years!
- `liverpoolborn` Sheila Jackson is an English WGM and the most successful player (highest win %).
- `meri_arabidze` Meri Arabidze has achieved the highest rating by a WGM of 2763 on chess.com.
- `sylvanaswindrunner` Jovana Rapport is currently rated 2712, very close to Meri's all time high!

Here I find these players in the `names` data to determine their indices.

"""

# ╔═╡ d59e5334-8373-11eb-28fb-a125951eb3e0
camillab = findfirst(i -> i == "camillab", names)

# ╔═╡ 4c8a3558-837b-11eb-246d-eb9502bcef0e
length(findall(g -> g== 4832, vec(games))) # number of games camillab played!

# ╔═╡ bdc5dc62-837a-11eb-028f-2b8f16915b4b
liverpoolborn = findfirst(i -> i == "liverpoolborn", names)

# ╔═╡ 29978e56-837b-11eb-3c8d-fb5bd59eb23c
meri_arabidze = findfirst(i -> i == "meri-arabidze", names)

# ╔═╡ f08c5240-8c12-11eb-3563-e908040de5af
sylvanaswindrunner = findfirst(i -> i == "sylvanaswindrunner", names)

# ╔═╡ 5bf8ab2c-837c-11eb-21b0-7d024db27d6d
md"""
Use the optimized variational approximation to give estimates for the following inference problem:

What is the probability player A is better than player B?

Recall from our early plots, this is the mass under the distribution in the region above the line of equal skill, $z_A > z_B$.


We have chosen a very simple variational family, factorized Gaussians. This would allow us to derive the exact expression for the probability under the variational approximation that player A has higher skill than player B. This year we will not derive this value exactly.

We can estimate the answer to our inference question by sampling $N = 10000$ samples from the variational distribution and using a Simple Monte Carlo estimate.

This inference question restricts us back to a two-player scenario. We know from previous questions that we can plot contours for distributions over 2 player skills.

#### most games vs most successful

1. Plot the **approximate** posterior over the skills of `camillab` and `liverpoolborn`.
2. Estimate the probability that `liverpoolborn` is more skillful than `camillab`.

#### all time high vs contender
3. Plot the **approximate** posterior over the skills of `meri_arabidze` and `sylvanaswindrunner`.
4. Estimate the probability that `sylvanaswindrunner` is more skillful than `meri_arabidze`.
"""

# ╔═╡ ee7bcf80-8c17-11eb-235f-75a9a8f619be
function find_μ_logσ(p1_idx, p2_idx, final_params)
	"""Given player indices, return their skill mean and log_std"""
	player_μ = [variational_params[1][p1_idx], variational_params[1][p2_idx]]
	player_logσ =  [variational_params[2][p1_idx], variational_params[2][p2_idx]]
	return player_μ, player_logσ
end

# ╔═╡ 5af2e5a0-8c12-11eb-18be-dd3606d0cd1e
function plot_var_approx(p1_idx, p2_idx, final_params)
	"""Plot the Variational Inference of 2-Player skills"""
	player_μ, player_logσ = find_μ_logσ(p1_idx, p2_idx, final_params)
	var_approx(zs) = exp.(log_prior(zs;μ=player_μ,logσ=player_logσ))
	p = plot(xlabel=names[p1_idx], ylabel=names[p2_idx])
	skillcontour!(var_approx, colour=:blue)
	plot_line_equal_skill!()
	return p
end

# ╔═╡ d1b07130-8c17-11eb-2716-9bd9e9ee0127
function prob_mc(p1_idx, p2_idx, final_params, num_samples=10000)
	"""Calculate the probability that Player 2 is more skillful than Player 1 using Monte-Carlo"""
	player_μ, player_logσ = find_μ_logσ(p1_idx, p2_idx, final_params)
	mc_samples = randn(2, num_samples) .* exp.(player_logσ) .+ player_μ
	skill_prob_mc = mean(mc_samples[1, :] .> mc_samples[2, :])
	return skill_prob_mc
end

# ╔═╡ 30855400-8c18-11eb-2b26-272905808c76
function prob_exact(p1_idx, p2_idx, final_params)
	"""Calculate the probability that Player 1 is more skillful than Player 2 using Exact method (to verify Monte-Carlo)"""
	player_μ, player_logσ = find_μ_logσ(p1_idx, p2_idx, final_params)
	dist_μ = player_μ[1] - player_μ[2]
	dist_σ = sqrt(sum(exp.(player_logσ).^2))
	skill_prob_exact = cdf(Normal(dist_μ, dist_σ), 0)
	return  1 - skill_prob_exact
end

# ╔═╡ 71cd6c00-8c12-11eb-22a3-712694d8ad6f
#1 Plot approximate posterior over liverpoolborn and camillab 
plot_var_approx(liverpoolborn, camillab, variational_params)

# ╔═╡ 8e04b30a-8380-11eb-0fbb-139af83c9332
#2 Use Simple Monte Carlo to estimate probability Liverpoolborn is better
prob_mc(liverpoolborn, camillab, variational_params)

# ╔═╡ 9a20298a-8380-11eb-3d05-5977357b0c04
#3 plot approximate posterior over sylvanaswindrunner and meri_arabidze
plot_var_approx(sylvanaswindrunner, meri_arabidze, variational_params)

# ╔═╡ 9cffc6a6-8380-11eb-069c-bf70f976758a
#4 Use Simple Monte Carlo to estimate probability sylvanaswindrunner is better
prob_mc(sylvanaswindrunner, meri_arabidze, variational_params)

# ╔═╡ Cell order:
# ╟─6c6b0aa6-82b2-11eb-3114-25412fb07e27
# ╟─225c993e-82b2-11eb-3322-31a522cc8594
# ╠═17feea80-8354-11eb-2662-094b31e3fe15
# ╠═4b5267e0-8a84-11eb-3107-c755af4d0e4d
# ╠═388c4d90-8ccf-11eb-1de4-cff652810053
# ╠═6497c1a0-8cd2-11eb-192a-4983db47d233
# ╠═6625b4a0-8cd2-11eb-010b-9db4b0c4d172
# ╠═8bf55322-8cd7-11eb-17e8-4b4428fe1c3e
# ╟─ce219a64-8350-11eb-37aa-1d156db00ce3
# ╠═b0113760-8b8d-11eb-27d1-f3d7edba2028
# ╠═237e7586-8351-11eb-0810-8b09c322cf0c
# ╠═a08b1960-8358-11eb-1d60-615634e45184
# ╠═cdc987cc-8358-11eb-118f-0f4d7ca2e040
# ╠═1392a8c4-8359-11eb-1639-29392237258c
# ╟─37c3b10e-8352-11eb-1c03-eddedcc516e0
# ╠═55c9ac2e-8381-11eb-2ee7-e564b39a2325
# ╠═5681e80e-835c-11eb-3e50-e32d41cb1ddb
# ╠═5617427e-835c-11eb-1ae3-ed0e13b24fe3
# ╟─e7f18c6e-835b-11eb-33ee-7de9b4f5071a
# ╠═f5e2fb76-8380-11eb-1a4b-1737bf6688bc
# ╠═f5a9cdd6-8380-11eb-07e1-d37f72198e39
# ╠═f56d321a-8380-11eb-3918-a7e25c3b654d
# ╠═d96c5102-8aaa-11eb-0131-3797c12c0ecc
# ╠═f4ec68ec-8380-11eb-2a55-e1ed7b634ae8
# ╟─08eb9304-835e-11eb-3798-ad7ecf016eb3
# ╠═3a7ce2aa-8382-11eb-2029-f97ac18d1f97
# ╠═09d97bd2-835e-11eb-2545-3b34186ea79d
# ╠═4491de6a-8382-11eb-3c4f-f3edbbd8b6d9
# ╟─a5b747c8-8363-11eb-0dcf-2311ebef4a2a
# ╠═a4890ef8-8382-11eb-0fa3-df568c42ef1e
# ╠═bc156e4a-8382-11eb-2fec-797f286ae96c
# ╠═bc160bac-8382-11eb-11be-57f2f3484921
# ╟─c1f56960-8cb4-11eb-0990-8d76d5ccbb91
# ╠═c1f495d0-8b92-11eb-385a-d59c7a75c1d7
# ╠═54573350-8365-11eb-2069-ff2041af30c5
# ╠═c064ce10-8b92-11eb-04ea-cb2a41b785c5
# ╠═5588835a-8365-11eb-01bf-dba9289b3f59
# ╠═21523b30-8b94-11eb-3a3d-bf0f3b875cb4
# ╠═56d3d372-8365-11eb-21f8-b941b23056e1
# ╠═bef092b0-8b94-11eb-28ff-932b0d0a13c0
# ╟─61baa2fa-8365-11eb-1d51-cb9ba987cb00
# ╠═615c1994-836f-11eb-3d3e-af2f4b7b7d80
# ╠═68d993cc-836f-11eb-2df1-c53cd5ea6ed4
# ╠═aa156070-8372-11eb-3d8c-4946a3e32d36
# ╟─4e914c4a-8369-11eb-369e-af5416995cb8
# ╠═f7513740-8cbb-11eb-1d33-a1d4f3f29b2b
# ╟─88de80c0-8cd8-11eb-2028-f1bf5fbff798
# ╠═6af25166-8384-11eb-24d9-85aed2d14ee2
# ╠═a34a1b64-8384-11eb-2e9f-0de11f1cfd4d
# ╠═b5dc532a-8384-11eb-244e-556e2769179e
# ╟─40321e00-8c0a-11eb-2ecf-c30844d8dc7a
# ╠═d19cc320-8c0c-11eb-3470-25dbd0bfbaeb
# ╠═efce7f36-8384-11eb-2499-19ffdb51451f
# ╟─f1e54852-836d-11eb-1847-13ab95054446
# ╠═d59e5334-8373-11eb-28fb-a125951eb3e0
# ╠═4c8a3558-837b-11eb-246d-eb9502bcef0e
# ╠═bdc5dc62-837a-11eb-028f-2b8f16915b4b
# ╠═29978e56-837b-11eb-3c8d-fb5bd59eb23c
# ╠═f08c5240-8c12-11eb-3563-e908040de5af
# ╟─5bf8ab2c-837c-11eb-21b0-7d024db27d6d
# ╠═ee7bcf80-8c17-11eb-235f-75a9a8f619be
# ╠═5af2e5a0-8c12-11eb-18be-dd3606d0cd1e
# ╠═d1b07130-8c17-11eb-2716-9bd9e9ee0127
# ╠═30855400-8c18-11eb-2b26-272905808c76
# ╠═71cd6c00-8c12-11eb-22a3-712694d8ad6f
# ╠═8e04b30a-8380-11eb-0fbb-139af83c9332
# ╠═9a20298a-8380-11eb-3d05-5977357b0c04
# ╠═9cffc6a6-8380-11eb-069c-bf70f976758a
