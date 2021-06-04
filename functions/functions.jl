using PyCall
pickle = pyimport("pickle")

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


function get_neighbor_map(D::Int, n::Int)
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


function pickle_func(dir, filename, z, obj)

	path = joinpath(dir, filename, "rawdata"*string(z)*".pkl")

	open(path, "w") do file
		pickle.dump(obj, file)
	end

	println("pickled!")
end


function prepare(step, m, n, Kb, J, itir, dims, dir, temperature)
	
	# check ferromagnaticy and add to filename
	if J < 1
		format = "anti"
	else 
		format = "normal"
	end

	# final filename
	dirname = format * "_" * string(dims) * "D_" * string(n) * "grid_" * string(itir) * "itir_" * string(step) * "step"
	text =
	"""
temperature steps: $step 
states per temperatures:  $m
grid length: $n
dimensions: $dims
Kb: $Kb
ferromagnetic $format
itirations $itir
saved as $dirname
temperature $temperature
	"""
	
	println("saving in ", dirname)

	try
		mkdir(joinpath(dir, dirname))
	catch err
		println("remove directory: ", joinpath(dir, dirname), " to continue")
		exit()
	end

	open(joinpath(dir, dirname, "config.txt"), "w") do file
		println(file, text)
	end

	return dirname
end