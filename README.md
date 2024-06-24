# Genetic Algorithm Matcher

## Overview

This project aims to solve a matching problem using a genetic algorithm. There are 30 men and 30 women, each with their preferences for members of the opposite gender. The goal is to find an optimal matching based on these preferences.

## Input

- **Preferences File**: A file named `GA_input.txt` containing preferences of men (first 30 lines) and women (next 30 lines). Each line represents the ranking preferences of a participant.

## Requirements

1. **Representation of Solutions**
   - Encoding potential solutions for the genetic algorithm.

2. **Evaluation Function**
   - Define a function to evaluate the quality of a solution.

3. **Crossover Implementation**
   - Perform crossover between two solutions to produce offspring.

4. **Mutation Implementation**
   - Introduce mutations in the solutions.

5. **Handling Early Convergence**
   - Strategies to prevent premature convergence to local optima.

6. **Execution File**
   - The program should read preferences from `GA_input.txt`, produce the optimal matching, and print the matching and its score.
   - The program should also generate data to create a report as specified.

## Report

1. **Explanation**
   - How the problem was represented.
   - How the optimal solution was defined.
   - Details of genetic operators (crossover and mutation) used.
   - Strategies to avoid local minima.
   - Structure of the evaluation function.

2. **Graphs**
   - Performance graphs showing the best, worst, and average evaluation function values over generations.

3. **Experiments**
   - Testing with different population sizes and number of generations (e.g., 180 individuals for 100 generations, 100 individuals for 180 generations, etc.).
   - Reporting the combination yielding the best solution with minimal evaluation function calls.

## Submission

- The code and an execution file that runs on either Windows or Linux (specify which).
- A report in Word or PDF format.
- Deadline: 27th June.
- Submissions can be done in pairs.

## Next Steps

1. **Design the Genetic Algorithm**
   - **Encoding**: Decide how to represent matchings.
   - **Evaluation Function**: Define how to measure the quality of a matching.
   - **Crossover**: Decide on a crossover method suitable for the matching problem.
   - **Mutation**: Implement mutation strategies to introduce diversity.
   - **Convergence**: Implement methods to prevent premature convergence.

2. **Implementation**
   - Write the code to implement the genetic algorithm.
   - Ensure the code reads from `GA_input.txt` and outputs the required results.

3. **Testing and Experimentation**
   - Run the algorithm with different parameter settings.
   - Collect data for the report.

4. **Documentation**
   - Prepare the report explaining your approach, findings, and graphs.