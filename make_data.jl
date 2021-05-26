using PyCall
pickle = pyimport("pickle")
include("functions/functions.jl")

function make_state(n)
	# change number of $n to change dimensions
	state = rand((-1,1), (n, n)) 
	return state
end
"""-------------------------------------------------------------------------------------"""

step 	= 100 	# number of temperatures
m 		= 10 	# number of states per temperature #TODO:add to name

n 		= 20 	# length of grid 

# Kb 		= 1.380649*10^-23 #Boltzman constant
Kb 		= 1 
J 		= 1 #if J negative: antiferromagnet

itir 	= 10^6 # number of itirations
dir 	= mkpath("./train_data")

Tk_dict = Dict(2 => 2.27, 3 => 4.5, 4 => 6.86) 


"""-------------------------------------------------------------------------------------"""

# gets dimensions 
dims = ndims(make_state(n))


# dependent on dimensions 
Tk = Tk_dict[dims]


# creates directory
filename = prepare(step, m, n, Kb, J, itir, dims, dir)


# makes linspace of temperature
temperature = LinRange(0, Tk*2, step) # |> collect


# makes a map of all possible neighbors (vectors)
neighbor_map = get_neighbor_map(dims, n)


big_list = []

# iterates the temperatue
for T in temperature
	
	println(T)
	KbT = Kb*T 

	# number of states per temperature
	for z in 1:m

		state = make_state(n)
		
		sub = []
		# push!(sub_final, [T, state])

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
			if (i-1)%100 == 0
				push!(sub, [T, state])
				# println(get_average(state))
			end

		end

		push!(big_list, sub)

	end

end

final = [dims, n, big_list]

pickle_func(dir, filename, final)
