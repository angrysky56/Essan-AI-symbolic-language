import random
import re
from typing import List, Tuple
import nltk
import spacy
from nltk.corpus import wordnet
import itertools
import matplotlib.pyplot as plt
import numpy as np


# Initialize NLTK and spaCy 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading en_core_web_ model for spaCy...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")




def get_synonyms(word):
    try:
        synonyms = []
        synsets = wordnet.synsets(word)
        if not synsets:
            return {word}

        for syn in synsets:
            if syn and syn.lemmas():
                for lemma in syn.lemmas():
                    if lemma and lemma.name():
                        synonyms.append(lemma.name())
        return set(synonyms) if synonyms else {word}
    
    except Exception as e:
        print(f"Error getting synonyms for {word}: {e}")
        return {word}  # Return original word if an error occurs

ESSAN_SYMBOL_KEYWORDS = {
    "⨿": ["essence", "core", "being", "entity", "concept", "identity"],
    "⧈": ["connect", "relate", "link", "join", "combine", "interact", "network", "relationship", "bond", "synergy"],
    "⧬": ["initiate", "begin", "start", "trigger", "activate", "originate", "commence", "launch", "introduce"],
    "⫰": ["move", "flow", "act", "transform", "change", "progress", "evolve", "shift", "transition", "dynamic", "adapt"],
    "⧿": ["cycle", "repeat", "recur", "iterate", "loop", "feedback", "reiterate", "oscillate"],
    "◬": ["change", "transform", "adapt", "evolve", "modify", "shift", "mutate", "alter", "vary", "adjust"],
    "⧉": ["strength", "amplify", "intensify", "enhance", "reinforce", "boost", "augment", "magnify", "increase", "power"],
    "⩘": ["declare", "confirm", "finalize", "complete", "conclude", "finish", "end", "resolve", "affirm", "verify"],
    "⩉": ["query", "inquire", "question", "explore", "investigate", "probe", "examine", "research", "seek", "discover"],
    "⍞": ["diminish", "limit", "reduce", "restrict", "constrain", "control", "moderate", "regulate", "contain"],
    "⧾": ["purpose", "intention", "goal", "objective", "aim", "target", "intent", "motive", "reason", "drive"],
    "║": ["boundary", "limit", "barrier", "constraint", "restriction", "border", "edge", "threshold", "fence", "wall"],
}



def calculate_task_accuracy(symbol_chain, task_description):
    """Evaluates semantic alignment using NLP."""
    try:
        doc = nlp(task_description.lower())
        task_elements = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

        symbol_relevance = 0
        for symbol in symbol_chain:
            if symbol in ESSAN_SYMBOL_KEYWORDS:
                keywords = ESSAN_SYMBOL_KEYWORDS[symbol]
                for keyword in keywords:
                    synonyms = get_synonyms(keyword)
                    if any(element in synonyms or element == keyword for element in task_elements):
                        symbol_relevance += 1
                        break

        return symbol_relevance / len(symbol_chain) if len(symbol_chain) > 0 else 0

    except Exception as e:
        print(f"Error in calculate_task_accuracy: {e}")
        return 0


def calculate_efficiency(symbol_chain):
    """Evaluates efficiency based on length."""
    return 1 / len(symbol_chain) if symbol_chain else 0

def calculate_adaptability(symbol_chain, original_task, modified_task):
    """Evaluates adaptability by comparing accuracy on different tasks."""
    original_accuracy = calculate_task_accuracy(symbol_chain, original_task)
    modified_accuracy = calculate_task_accuracy(symbol_chain, modified_task)
    return (original_accuracy + modified_accuracy) / 2

def calculate_recursive_depth(symbol_chain):
    """Evaluates presence of recursive symbols."""
    return min(symbol_chain.count("⧿"), 1)  # Max score of 1 for recursion

def calculate_user_satisfaction(symbol_chain, task_description):
    """Simulates user satisfaction (can be replaced with actual feedback)."""
    accuracy = calculate_task_accuracy(symbol_chain, task_description)
    length_preference = 0.9 if len(symbol_chain) < 15 else 0.7
    recursion_present = 0.95 if "⧿" in symbol_chain else 0.8
    diversity = len(set(symbol_chain)) / len(symbol_chain) if symbol_chain else 0  # Symbol diversity

    satisfaction = (0.5 * accuracy +
                    0.2 * length_preference +
                    0.2 * recursion_present +
                    0.1 * diversity)

    satisfaction += random.uniform(-0.1, 0.1) # Add some randomness
    return max(0, min(1, satisfaction)) # Keep score between 0 and 1



def calculate_fitness(symbol_chain, task_description, modified_task_description, weights):
    """Calculates the overall fitness score."""

    accuracy = calculate_task_accuracy(symbol_chain, task_description)
    efficiency = calculate_efficiency(symbol_chain)
    adaptability = calculate_adaptability(symbol_chain, task_description, modified_task_description)
    recursion = calculate_recursive_depth(symbol_chain)
    user_satisfaction = calculate_user_satisfaction(symbol_chain, task_description)


    fitness = (weights["accuracy"] * accuracy +
               weights["efficiency"] * efficiency +
               weights["adaptability"] * adaptability +
               weights["recursion"] * recursion +
               weights["user_satisfaction"] * user_satisfaction)

    return fitness

def generate_initial_population(population_size, symbol_set, min_length=5, max_length=20):
    """Generates the initial population of symbol chains."""

    population = []
    for _ in range(population_size):
        chain_length = random.randint(min_length, max_length)
        symbol_chain = "".join(random.choice(symbol_set) for _ in range(chain_length))
        population.append(symbol_chain)
    return population

def crossover(parent1, parent2):
    """Basic crossover operation (fallback)."""
    if not parent1 or not parent2:
        return "", "" # return better defaults to prevent concatenation issues

    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def contextual_crossover(parent1, parent2, task_description):
    """Performs crossover at semantically similar points based on task description."""

    doc = nlp(task_description.lower())
    
    best_similarity = -1 
    best_crossover_point_parent1 = None
    best_crossover_point_parent2 = None

    for i in range(1, len(parent1)):
        segment1 = parent1[:i]
        for j in range(1, len(parent2)):
            segment2 = parent2[:j]
            
            if segment1.strip() and segment2.strip():
                combined_segment = segment1 + ' ' + segment2
                if combined_segment.strip() and len(combined_segment.split()) > 2:
                    combined_doc = nlp(combined_segment)
                    if combined_doc.vector_norm > 0:
                        sim = doc.similarity(combined_doc)
                    else:
                        sim = -1  # Assign a low similarity score if combined_doc has an empty vector
                else:
                    sim = -1  # Assign a low similarity score if the combined segment is empty or not meaningful enough
            else:
                sim = -1  # Assign a low similarity score to avoid empty segments

            if sim > best_similarity:
                best_similarity = sim
                best_crossover_point_parent1 = i
                best_crossover_point_parent2 = j

    if best_crossover_point_parent1 is not None and best_crossover_point_parent2 is not None:
        child1 = parent1[:best_crossover_point_parent1] + parent2[best_crossover_point_parent2:]
        child2 = parent2[:best_crossover_point_parent2] + parent1[best_crossover_point_parent1:]
        return child1, child2

    else:  # Fallback: basic crossover if no segments increase similarity
        return crossover(parent1, parent2)




def mutate(symbol_chain, symbol_set, mutation_rate=0.1):
    """Introduces random mutations."""
    mutated_chain = ""
    for symbol in symbol_chain:
        if random.random() < mutation_rate:
            mutated_chain += random.choice(symbol_set)
        else:
            mutated_chain += symbol
    return mutated_chain



def adaptive_mutation(symbol_chain, symbol_set, fitness, mutation_rate_range=(0.05, 0.2)):
    """Adapts the mutation rate based on fitness."""
    min_rate, max_rate = mutation_rate_range
    normalized_fitness = max(0, min(1, fitness))  # Ensure fitness is in [0, 1]
    mutation_rate = max_rate - (normalized_fitness * (max_rate - min_rate))
    return mutate(symbol_chain, symbol_set, mutation_rate)



def select_parents(population, fitnesses, num_parents=2):
    """Selects parents using tournament selection."""

    if not population or not fitnesses or len(population) < num_parents:
        return []  # Return empty list if invalid input
    
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(range(len(population)), k=min(3, len(population)))
        winner = tournament[0]
        for i in tournament[1:]:
            if fitnesses[i] > fitnesses[winner]:
                winner = i
        parents.append(population[winner])

    return parents


def run_genetic_algorithm(task_description, modified_task_description, weights, 
                         population_size=50, num_generations=100, 
                         symbol_set=None, mutation_rate_range=(0.05, 0.2)):

    if symbol_set is None:
        symbol_set = list(ESSAN_SYMBOL_KEYWORDS.keys())

    population = generate_initial_population(population_size, symbol_set)
    
    for generation in range(num_generations):
        fitnesses = [calculate_fitness(chain, task_description, modified_task_description, weights) 
                     for chain in population]

        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitnesses)

            if parents and len(parents) >= 2:  # Ensure enough parents for crossover
                children = contextual_crossover(parents[0], parents[1], task_description)
                for child in children:
                    
                    child_fitness = calculate_fitness(child, task_description, modified_task_description, weights)
                    mutated_child = adaptive_mutation(child, symbol_set, child_fitness, mutation_rate_range)
                    new_population.append(mutated_child)


        if new_population:
            population = new_population


    fitnesses = [calculate_fitness(chain, task_description, modified_task_description, weights) for chain in population]

    if fitnesses:
        best_chain = population[fitnesses.index(max(fitnesses))]
        return best_chain
    else:
        return None # Explicitly return None if no chains generated

def experiment(task_description, modified_task_description, weights, param_combinations):
    results = []
    for params in param_combinations:
        pop_size, num_gens, mut_rate = params
        best_chain = run_genetic_algorithm(task_description,  modified_task_description,  weights,  pop_size,  num_gens, mutation_rate_range=(0.05, mut_rate))

        if best_chain:
            best_fitness = calculate_fitness(best_chain, task_description, modified_task_description, weights)
            results.append((params, best_fitness))
    return results




def plot_avg_vs_param(results, param_index, param_name, task_name):
    param_values = sorted(list(set([params[param_index] for params, _ in results])))
    avg_fitness_by_param = []

    for value in param_values:
        fitnesses = [fitness for params, fitness in results if params[param_index] == value]
        avg_fitness = np.mean(fitnesses) if fitnesses else 0
        avg_fitness_by_param.append(avg_fitness)


    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
    plt.plot(param_values, avg_fitness_by_param, marker='o', linestyle='-')
    plt.title(f"Average Fitness vs. {param_name}\n({task_name})", fontsize=14)  # Increase title font size
    plt.xlabel(param_name, fontsize=12)  # Increase label font size
    plt.ylabel("Average Fitness", fontsize=12)  # Increase label font size
    plt.xticks(fontsize=10)   # Increase tick font size
    plt.yticks(fontsize=10)   # Increase tick font size    
    plt.grid(True)
    plt.show()


def plot_heatmap(results, task_name, param1_index, param2_index, param1_name, param2_name):
    param1_values = sorted(list(set([params[param1_index] for params, _ in results])))
    param2_values = sorted(list(set([params[param2_index] for params, _ in results])))

    fitness_matrix = np.zeros((len(param1_values), len(param2_values)))

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            fitnesses = [fitness for params, fitness in results 
                        if params[param1_index] == val1 and params[param2_index] == val2]
            avg_fitness = np.mean(fitnesses) if fitnesses else 0
            fitness_matrix[i, j] = avg_fitness

    plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
    plt.imshow(fitness_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Fitness')
    plt.title(f"Average Fitness: {param1_name} vs. {param2_name}\n({task_name})", fontsize=14)  # Larger title
    plt.xticks(np.arange(len(param2_values)), param2_values, fontsize=10)  # Larger ticks
    plt.yticks(np.arange(len(param1_values)), param1_values, fontsize=10)
    plt.xlabel(param2_name, fontsize=12) # Larger label
    plt.ylabel(param1_name, fontsize=12) # Larger label
    plt.show()

# Main execution block
if __name__ == "__main__":

    tasks = {
        "Analyze sales data.": (
            "Analyze revised sales data.", 
            {"accuracy": 0.4, "efficiency": 0.3, "adaptability": 0.15, "recursion": 0.05, "user_satisfaction": 0.1}
        ),
        "Learn from user feedback and adapt the interface.": (
            "Learn from developer feedback and adapt the backend systems.", 
            {"accuracy": 0.2, "efficiency": 0.1, "adaptability": 0.4, "recursion": 0.2, "user_satisfaction": 0.1}
        ),
        "Initiate a project, gather resources, execute tasks, and evaluate results.": (
            "Initiate a study, gather participants, analyze data, and publish findings.", 
            {"accuracy": 0.3, "efficiency": 0.15, "adaptability": 0.25, "recursion": 0.2, "user_satisfaction": 0.1}
        ),
    }

    population_sizes = [50, 75, 100]
    num_generations = [50, 75, 100]
    mutation_rates = [0.05, 0.1, 0.15]
    param_combinations = list(itertools.product(population_sizes, num_generations, mutation_rates))

    all_results = {}
    for task, (modified_task, weights) in tasks.items():
        print(f"\nProcessing task: {task}")
        results = experiment(task, modified_task, weights, param_combinations)
        all_results[task] = results

        for params, best_fitness in results:
            print(f"    Parameters: {params}, Best Fitness: {best_fitness:.4f}")

    # Visualizations 
    for task_name, results in all_results.items():
        print(f"\nGenerating visualizations for task: '{task_name}'")
        plot_avg_vs_param(results, 0, "Population Size", task_name)
        plot_avg_vs_param(results, 1, "Number of Generations", task_name)
        plot_avg_vs_param(results, 2, "Mutation Rate", task_name)
        plot_heatmap(results, task_name, 0, 2, "Population Size", "Mutation Rate")
