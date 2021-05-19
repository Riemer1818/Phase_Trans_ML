using Plots

function get_average(A::Array)
	# gets average of a matrix / array
	E = 0
	D = 1

	# calculates the total by indicing the array and summing
	# TODO: use sum() function instead?
	for i in eachindex(A)
		E += A[i]
	end

	# multiplies dimentions to get number of objects in the array
	for d in size(A)
		D *= d
	end

	# check https://www.juliabloggers.com/basics-of-generating-random-numbers-in-julia/ 
	average = E/D
	return average
end


function get_neighbor_map(A::Array, D::Int, n::Int)
	# get's all the cord locations of the neighbors dependent on dimensions

	M = Array[]

	# iterates the dimensions
	for d in 1:D

		# creates maps eg [0,0,0,1] and [0,0,0,(n-1)]. Only sees direct neighbors 
		for n in [1,(n-1)]
			a = zeros(Int64, D)
			a[d] = a[d] + n 
			push!(M, a)
		end

	end

	return M

end


function get_particle_energy(A::Array, M::Array, n::Int, random_locat::Array, J::Int)
	# gets the energy of a specific particle. Looks at all neighbors in M

	# sets energy as current energy
	part_spin = A[CartesianIndex(Tuple(random_locat))] #(-1 | 1)

	H = 0

	# indices all neighbors
	for x in eachindex(M)

		# gets neighbors location
		neighbor = random_locat + M[x]

		# fixes edge cases
		neighbor = ((neighbor.-1) .% n) .+ 1

		# adds energy to part energy
		H += A[CartesianIndex(Tuple(neighbor))]*part_spin

	end

	# returns the particle energy 
	return J*H*2 

end

# define J


function get_random_particle(dims)
	# gets a random partile location in grid

	random_locat = Int64[]

	# gets a random number for each dimension from 1:n
	for d in 1:dims
		push!(random_locat, rand(1:n))
	end
	
	return random_locat

end


function calc_flip(∆H::Int64, KbT::Float64)
	# returns bool dependent on difference in energy. 
	particle_energy = 0

	# when energy decreases, the state is switched (flipped), from just looking at neighbors
	if ∆H <= 0
		return true
	
	else
		# this is unclear in the python file
		particle_energy = exp(-∆H/KbT)

		# compares to uniform random float 0 - 1, random chance for the particle to flip 
		if particle_energy >= rand()
			return true

		else 
			return false

		end

	end

end


# These you should play with
step 	= 100 	# number of temperatures
m 		= 1		# number of states per temperature
n 		= 20 	# length of grid 

# Kb 		= 1.380649*10^-23 #Boltzman constant
Kb 		= 1 # why does this work? 
J 		= 1 #if J negative: antiferromagnet
itir 	= 10^4

# gets random state matrix 
state = rand((-1,1), (n, n)) 

# defines number of dimensions
dims = ndims(state)


if dims == 2
	Tk = 2.27	# Critical temperature for 2D
elseif dims == 3
	Tk = 4.5	# Critical temperatrue for 3D
elseif dims == 4
	Tk 	= 6.86 	# Critical temperature for 4D
end


# check ferromagnaticy and add to filename
if J < 1
	format = "anti"
else 
	format = "normal"
end

# final filename
plotname = format * "_" * string(dims) * "D_" * string(n) * "grid_" * string(itir) * "itir_" * string(step) * "step.png"
println(plotname)

# makes linspace of temperature
temperature = LinRange(0, Tk*2, step)  |> collect
# println(temperature)

# makes a map of all possible neighbors (vectors)
neighbor_map = get_neighbor_map(state, dims, n)

final = []

# iterates the temperatue
for T in temperature
	
	println(T)
	KbT = Kb*T
	sub_list = [] 

	# number of states per temperature
	for z in 1:m

		state = rand((-1,1), (n, n))

		push!(sub_list, state)

		for i in 1:itir
			# gets a random particle cartesian index 
			random_locat = get_random_particle(dims)
			# println(random_locat)

			# gets energy of this random particle 
			∆E = get_particle_energy(state, neighbor_map, n, random_locat, J)
			# println(∆E) 

			# calculates if (random) particle flips 
			if calc_flip(∆E, KbT)
				state[CartesianIndex(Tuple(random_locat))] *= -1
				# println(CartesianIndex(Tuple(random_locat)))
			end

			# every 100 states a state is saved
			if i%100 == 0
				push!(sub_list, state)
				# println(get_average(state))
			end

			# at the very last state, the average spin is printed
			if i == itir
				# println("state average ", get_average(state))
				push!(final, abs(get_average(state)))

			end

		end

	end

end

savefig(plot(temperature, final), plotname)

