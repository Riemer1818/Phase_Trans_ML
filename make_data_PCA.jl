using PyCall
include("functions/functions.jl")

function make_state(n)
	# change number of $n to change dimensions
	state = rand((-1,1), (n, n)) 
	return state
end

"""-------------------------------------------------------------------------------------"""

# step 	= 100 	# number of temperatures
step 	= parse(Int64, ARGS[1])

# n 		= 20 	# length of grid 
n 		= parse(Int64, ARGS[2])

# Kb 		= 1.380649*10^-23 #Boltzman constant
Kb 		= 1 

# J 		= 1 #if J negative: antiferromagnet
J 		= parse(Int64, ARGS[3])

# number of itirations per atom on average
it_p_atom		= parse(Int64, ARGS[4])

# dir 	= mkpath("./train_data")
dir 	= mkpath(ARGS[5])

Tk_dict = Dict(2 => 2.27, 3 => 4.5, 4 => 6.86) 


"""-------------------------------------------------------------------------------------"""

# gets dimensions 
dims = ndims(make_state(n))

# dependent on dimensions 
Tk = Tk_dict[dims]

# makes linspace of temperature
temperature = LinRange(2.6, 2.2, step) # |> collect

# number of itirations 
itir = n^dims * it_p_atom 

# creates directory
filename = prepare(step, n, Kb, J, it_p_atom, dims, dir, temperature)

# makes a map of all possible neighbors (vectors)
neighbor_map = get_neighbor_map(dims, n)

final = []

state = make_state(n)

# iterates the temperatue
final = []

for T in temperature

	for z in 1:1000
		sub = []
		
		KbT = Kb*T 		

		for i in 1:itir
			# gets a random particle cartesian index 
			random_locat = get_random_particle(dims)

			# gets energy of this random particle 
			∆E = get_particle_energy(state, neighbor_map, n, random_locat, J)

			# calculates if (random) particle flips 
			if calc_flip(∆E, KbT)
				state[CartesianIndex(Tuple(random_locat))] *= -1
			end

		end

		sub = [T, deepcopy(state)]
		push!(final, sub)	

	end
end

pickle_func(dir, filename, 1, final)
