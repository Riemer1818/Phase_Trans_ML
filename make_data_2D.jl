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

# number of states saved per temperature
step2 	= parse(Int64, ARGS[2])

# m 		= 10 	# number of states per temperature 
m 		= parse(Int64, ARGS[3])

# n 		= 20 	# length of grid 
n 		= parse(Int64, ARGS[4])

# Kb 		= 1.380649*10^-23 #Boltzman constant
Kb 		= 1 

# J 		= 1 #if J negative: antiferromagnet
J 		= parse(Int64, ARGS[5])

# number of itirations per atom on average
it_p_atom		= parse(Int64, ARGS[6])


# dir 	= mkpath("./train_data")
dir 	= mkpath(ARGS[7])

Tk_dict = Dict(2 => 2.27, 3 => 4.5, 4 => 6.86) 


"""-------------------------------------------------------------------------------------"""

# gets dimensions 
dims = ndims(make_state(n))

# dependent on dimensions 
Tk = Tk_dict[dims]

# makes linspace of temperature
temperature = LinRange(0, 2*Tk, step) # |> collect

# number of itirations 
itir = n^dims * it_p_atom

# creates directory
filename = prepare(step, m, n, Kb, J, it_p_atom, dims, dir, temperature)

# get's 100 values equally distributed over last 50% of itirations 
save_steps = range(itir/2, stop = itir, length=step2) 
# save_steps = range(1, stop=itir, length=100) 

floor_save_steps = []
for s_step in save_steps
	push!(floor_save_steps, floor(s_step))
end

# makes a map of all possible neighbors (vectors)
neighbor_map = get_neighbor_map(dims, n)


# number of states per temperature
for z in 1:m
	
	println(z)

	sub = []

	# iterates the temperatue
	for T in temperature
		
		KbT = Kb*T 		

		state = make_state(n)

		for i in 1:itir
			# gets a random particle cartesian index 
			random_locat = get_random_particle(dims)

			# gets energy of this random particle 
			∆E = get_particle_energy(state, neighbor_map, n, random_locat, J)

			# calculates if (random) particle flips 
			if calc_flip(∆E, KbT)
				state[CartesianIndex(Tuple(random_locat))] *= -1
			end

			# every 100 states a state is saved
			if i in floor_save_steps
				push!(sub, [T, deepcopy(state)])
			end
		end
	end
	final = [dims, n, sub]
	pickle_func(dir, filename, z, final)

end