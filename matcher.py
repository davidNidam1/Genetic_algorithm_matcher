import numpy as np
import random

def read_preferences(file_path):
    with open(file_path, 'r') as file:
        preferences = np.array([list(map(int, line.split())) for line in file])
    return preferences[:30], preferences[30:]

def evaluate_solution(solution, men_preferences, women_preferences):
    score = 0
    for man, woman in enumerate(solution):
        man_pref = np.where(men_preferences[man] == woman + 1)[0][0]
        woman_pref = np.where(women_preferences[woman] == man + 1)[0][0]
        score += man_pref + woman_pref
    return score

def crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = [0]*size, [0]*size

    c, d = sorted([random.randint(0, size) for _ in range(2)])
    for i in range(c, d):
        p1[parent1[i]] = True
        p2[parent2[i]] = True

    child1, child2 = [-1]*size, [-1]*size
    child1[c:d], child2[c:d] = parent1[c:d], parent2[c:d]

    for i in range(size):
        if not p1[parent2[i]]:
            child1[child1.index(-1)] = parent2[i]
        if not p2[parent1[i]]:
            child2[child2.index(-1)] = parent1[i]

    return child1, child2

def mutate(solution):
    size = len(solution)
    p1, p2 = random.randint(0, size - 1), random.randint(0, size - 1)
    solution[p1], solution[p2] = solution[p2], solution[p1]

def genetic_algorithm(men_preferences, women_preferences, population_size=100, generations=1000):
    population = [random.sample(range(30), 30) for _ in range(population_size)]
    best_solution = min(population, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))
    best_score = evaluate_solution(best_solution, men_preferences, women_preferences)

    for generation in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = sorted(new_population, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))[:population_size]

        current_best = min(population, key=lambda sol: evaluate_solution(sol, men_preferences, women_preferences))
        current_score = evaluate_solution(current_best, men_preferences, women_preferences)
        
        if current_score < best_score:
            best_solution, best_score = current_best, current_score

    return best_solution, best_score

if __name__ == "__main__":
    men_preferences, women_preferences = read_preferences('GA_input.txt')
    best_solution, best_score = genetic_algorithm(men_preferences, women_preferences)
    print("Best Matching:", best_solution)
    print("Best Score:", best_score)
