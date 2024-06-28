import numpy as np
import random
import matplotlib.pyplot as plt

# Function to read preferences from a file and split into men and women preferences
def read_preferences(file_path):
    with open(file_path, 'r') as file:
        preferences = np.array([list(map(int, line.split())) for line in file])
    return preferences[:30], preferences[30:]

# Function to evaluate a solution by calculating the sum of ranks of partners in their preferences
def evaluate_solution(solution, men_preferences, women_preferences):
    score = 0
    for man, woman in enumerate(solution):
        man_pref = np.where(men_preferences[man] == woman + 1)[0][0]
        woman_pref = np.where(women_preferences[woman] == man + 1)[0][0]
        score += man_pref + woman_pref
    return score

# Function to get the best and worst possible scores for normalization
def get_score_range(men_preferences, women_preferences):
    worst_score = 0
    best_score = 0
    n = len(men_preferences)
    for i in range(n):
        worst_score += n - 1  # Worst rank is (n-1) for 0-based index
        best_score += 0       # Best rank is 0 for 0-based index
    return best_score, worst_score

# Function to normalize scores to a range of 1-100
def normalize_score(score, best_score, worst_score):
    return 100 * (1 - (score - best_score) / (worst_score - best_score))

# Function to perform order crossover between two parents
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child1 = [-1] * size
    child2 = [-1] * size
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    fill_order(child1, parent2, start, end)
    fill_order(child2, parent1, start, end)
    return child1, child2

# Helper function to fill the remainder of the child in order crossover
def fill_order(child, parent, start, end):
    size = len(child)
    current_pos = end % size
    for i in range(end, end + size):
        pos = i % size
        if parent[pos] not in child:
            while child[current_pos] != -1:
                current_pos = (current_pos + 1) % size
            child[current_pos] = parent[pos]

# Function to perform cycle crossover between two parents
def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size
    cycle = 0
    while -1 in child1:
        if cycle % 2 == 0:
            index = child1.index(-1)
            while child1[index] == -1:
                child1[index] = parent1[index]
                index = parent1.index(parent2[index])
        else:
            index = child1.index(-1)
            while child1[index] == -1:
                child1[index] = parent2[index]
                index = parent1.index(parent2[index])
        cycle += 1
    cycle = 0
    while -1 in child2:
        if cycle % 2 == 0:
            index = child2.index(-1)
            while child2[index] == -1:
                child2[index] = parent2[index]
                index = parent2.index(parent1[index])
        else:
            index = child2.index(-1)
            while child2[index] == -1:
                child2[index] = parent1[index]
                index = parent2.index(parent1[index])
        cycle += 1
    return child1, child2

# Function to mutate a solution by swapping two random positions
def mutate(solution, mutation_rate):
    size = len(solution)
    for _ in range(int(size * mutation_rate)):
        p1, p2 = random.randint(0, size - 1), random.randint(0, size - 1)
        solution[p1], solution[p2] = solution[p2], solution[p1]

# Function to select a parent using tournament selection
def tournament_selection(population, men_preferences, women_preferences, tournament_size=3):
    selected = random.sample(population, tournament_size)
    selected = sorted(selected, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))
    return selected[0]

# Function to apply local search to refine a solution
def local_search(solution, men_preferences, women_preferences, max_iter=10):
    best_solution = solution.copy()
    best_score = evaluate_solution(best_solution, men_preferences, women_preferences)
    for _ in range(max_iter):
        neighbor = solution.copy()
        mutate(neighbor, mutation_rate=0.1)
        neighbor_score = evaluate_solution(neighbor, men_preferences, women_preferences)
        if neighbor_score < best_score:
            best_solution = neighbor
            best_score = neighbor_score
    return best_solution

# Function to apply fitness sharing for maintaining diversity in the population
def fitness_sharing(population, men_preferences, women_preferences):
    niche_radius = 5  # Define a niche radius
    niche_penalty = 0.1  # Define a niche penalty
    for i, sol1 in enumerate(population):
        niche_count = 1
        for sol2 in population:
            if sol1 != sol2:
                distance = sum(1 for x, y in zip(sol1, sol2) if x != y)
                if distance < niche_radius:
                    niche_count += 1
        niche_factor = 1 / niche_count
        population[i] = (sol1, evaluate_solution(sol1, men_preferences, women_preferences) * niche_factor)
    return [sol for sol, _ in sorted(population, key=lambda x: x[1])]

# Main function implementing the genetic algorithm
def genetic_algorithm(men_preferences, women_preferences, population_size, generations, initial_mutation_rate=0.05, elitism_count=5, tournament_size=5):
    population = [random.sample(range(30), 30) for _ in range(population_size)]
    best_solution = min(population, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))
    best_score = evaluate_solution(best_solution, men_preferences, women_preferences)
    
    best_scores = []
    worst_scores = []
    average_scores = []

    best_possible_score, worst_possible_score = get_score_range(men_preferences, women_preferences)

    for generation in range(generations):
        new_population = sorted(population, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))[:elitism_count]

        mutation_rate = initial_mutation_rate * (1 - generation / generations)
        
        # Increase mutation rate if early convergence is detected
        if generation > 20 and len(set(best_scores[-10:])) == 1:
            mutation_rate *= 2

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, men_preferences, women_preferences, tournament_size)
            parent2 = tournament_selection(population, men_preferences, women_preferences, tournament_size)
            
            if random.random() < 0.5:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = cycle_crossover(parent1, parent2)
            
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])

        # Fitness sharing to maintain diversity
        new_population = fitness_sharing(new_population, men_preferences, women_preferences)

        # Re-introduce random individuals to maintain diversity
        for _ in range(int(population_size * 0.05)):
            new_population[random.randint(0, population_size - 1)] = random.sample(range(30), 30)

        # Apply local search to improve solutions
        for i in range(elitism_count, len(new_population), population_size // 10):
            new_population[i] = local_search(new_population[i], men_preferences, women_preferences)

        population = new_population[:population_size]

        scores = [evaluate_solution(sol, men_preferences, women_preferences) for sol in population]
        current_best = min(scores)
        current_worst = max(scores)
        current_average = sum(scores) / len(scores)

        best_scores.append(normalize_score(current_best, best_possible_score, worst_possible_score))
        worst_scores.append(normalize_score(current_worst, best_possible_score, worst_possible_score))
        average_scores.append(normalize_score(current_average, best_possible_score, worst_possible_score))

        if current_best < best_score:
            best_solution = population[scores.index(current_best)]
            best_score = current_best

    return best_solution, best_score, best_scores, worst_scores, average_scores

# Function to plot the results
def plot_results(generations, best_scores, worst_scores, average_scores, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), best_scores, label='Best Score')
    plt.plot(range(generations), worst_scores, label='Worst Score')
    plt.plot(range(generations), average_scores, label='Average Score')
    plt.ylim(0, 100)
    plt.xlabel('Generations')
    plt.ylabel('Normalized Score (1-100)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    men_preferences, women_preferences = read_preferences('GA_input.txt')
    
    # Test combinations of population size and generations
    combinations = [
        (100, 180),  # Population size = 100, Generations = 180
        (180, 100),  # Population size = 180, Generations = 100
        (150, 120)   # Population size = 150, Generations = 120
    ]
    
    for population_size, generations in combinations:
        best_solution, best_score, best_scores, worst_scores, average_scores = genetic_algorithm(
            men_preferences, women_preferences, population_size, generations, initial_mutation_rate=0.05, elitism_count=5, tournament_size=5)
        
        best_possible_score, worst_possible_score = get_score_range(men_preferences, women_preferences)
        normalized_score = normalize_score(best_score, best_possible_score, worst_possible_score)
        
        print(f"Combination: Population Size = {population_size}, Generations = {generations}")
        print("Best Matching:", best_solution)
        print("Best Score:", best_score)
        print("Normalized Score (1-100):", normalized_score)
        
        title = f"Population Size: {population_size}, Generations: {generations}"
        plot_results(generations, best_scores, worst_scores, average_scores, title)
