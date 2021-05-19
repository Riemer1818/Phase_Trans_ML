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

function get_particle_energy(A::Array, M::Array, n::Int, random_locat::Array)
	
	# sets energy as current energy
	part_energy = A[CartesianIndex(Tuple(random_locat))]


	# indices all neighbors
	for x in eachindex(M)

		# gets neighbors location
		locat = random_locat + M[x]

		# fixes edge cases
		locat = ((locat.-1) .% n) .+ 1

		# adds energy to part energy
		part_energy *= -A[CartesianIndex(Tuple(locat))] 

	end

	# returns the particle energy *-2 (why *-2?)
	return (part_energy*-2)

end


function get_random_particle(dims)
	# gets a random partile location in grid

	random_locat = Int64[]

	# gets a random number for each dimension from 1:n
	for d in 1:dims
		push!(random_locat, rand(1:n))
	end
	
	return random_locat

end


function calc_flip(∆E::Int64, KbT::Float64, random_locat::Array)
	particle_energy = 0

	# when energy decreases, the state is switched (flipped), from just looking at neighbors
	if ∆E <= 0
		state[CartesianIndex(Tuple(random_locat))] *= -1
		particle_flip = true
	else
		# this is unclear in the python file
		partile_energy = exp(-∆E/KbT)

		# compares to uniform random float 0 - 1, random chance for the particle to flip 
		if particle_energy >= rand()

			# replaces the particle in the state 
			state[CartesianIndex(Tuple(random_locat))] *= -1

			particle_flip = true

		else 
			particle_flip = false
		end
	end

	return particle_flip

end


Tk 		= 2.27	# Critical temperature
step 	= 100 	# number of temperatures
m 		= 5 	# number of states per temperature
n 		= 50 	# length of grid 


# gets random state matrix 
state = rand((-1,1), (n, n, n))
# state = [1 2 3; 4 5 6; 7 8 9]

# defines number of dimensions
dims = ndims(state)

# makes linspace of temperature
temperature = LinRange(0.001, Tk*2, step)  |> collect
# println(temperature)

# makes a map of all possible neighbors (vectors)
neighbor_map = get_neighbor_map(state, dims, n)

final = []

# iterates the temperatue
for KbT in temperature
	println(KbT)

	# number of states per temperature
	for z in 1:m
		# gets a random particle cartesian index 
		random_locat = get_random_particle(dims)

		# gets energy of this random particle 
		∆E = get_particle_energy(state, neighbor_map, n, random_locat)
		# println(∆E)

		# calculates if (random) particle flips 
		if calc_flip(∆E, KbT, random_locat)
			state[random_locat] *= -1
			println("something flipped")
		end

	end
	# println(state)
	println(get_average(state))
	final.push!(sub_list)

end
