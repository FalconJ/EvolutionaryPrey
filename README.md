# Predator Prey Evolution 

Implemention of an algorithm for multiobjective optimization with at least two objectives. I decided to implement the Predator Prey algorithm.
I decided on this algorithm because I was interested to see how a small implementation without much complexity would turn out, in this case I followed this algorithm.

### begin


	**INITIALIZE** prey and predator population in a lattice
	**EVALUATE** each individual in the prey population
	
	*repeate*
		**MOVE** each prey one space in the lattice
		**SELECT** parents for the prey by looking for neighbors
		**RECOMBINE** pair of parents		
		**MUTATE** the entire prey population with a chance	
		**CALCULATE** number of hunts for the predators
		*repeate*
			**IF** any prey IS neighbor of a predator
				**KILL** the least fit prey
				**MOVE** the predator to the prey space
			**ELSE**
				**MOVE** the predator in a random direction
		*until* number of hunts is reached
	*until* amount of generations is reached
### end
