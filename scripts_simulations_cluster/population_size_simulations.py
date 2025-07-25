import numpy as np
from scipy.stats import poisson, lognorm
import csv
import os
from mpi4py import MPI

# Bacteria parameters
bS = 1.7        # birth rate sensitive
dS = 1      # death rate sensitive
cS = 1        # competition rate sensitive (no competition)

# Persister parameters (very low birth and death rates)
bP = 0.0      # birth rate for persisters
dP = 0.1      # death rate for persisters

# General parameters
t0 = 0        # initial time
mu = 0         # mean of the associated normal distribution

def mic_to_br(mic, min_bR=2.0):
    max_bR = 2.0
    min_mic = 1
    max_mic = 32
    bR = max_bR - (max_bR - min_bR) * (mic - min_mic) / (max_mic - min_mic)
    return bR

def mic_mapping(mic, min_value=-0.5, min_mic_threshold=6.25):
    """
    Maps MIC to a value based on the min_mic_threshold.
    If mic >= min_mic_threshold, returns 0.
    Otherwise, returns a value between min_value and 0 based on linear scaling.
    """
    max_value = 0.0
    min_mic = 1
    max_mic = min_mic_threshold

    if mic >= min_mic_threshold:
        return 0

    mapped_value = min_value + (max_value - min_value) * (mic - min_mic) / (max_mic - min_mic)
    return mapped_value

def generate_intervals(growth_duration, treatment_duration, total_time, treatment_concentration):
    intervals = []
    t = 0
    while t < total_time:
        # Treatment phase
        if t + treatment_duration <= total_time:
            intervals.append((t, t + treatment_duration, treatment_concentration))
        else:
            intervals.append((t, total_time, treatment_concentration))
            break

        # Growth phase
        if t + treatment_duration + growth_duration <= total_time:
            intervals.append((t + treatment_duration, t + treatment_duration + growth_duration, 0))
        else:
            intervals.append((t + treatment_duration, total_time, 0))
            break

        t += growth_duration + treatment_duration

    return intervals

def RUN_tau_leaping(tfin, S_MIC, intervals, initial_pers_level, mu, sigma, K, initial_pop_size, tau_max=0.1,
                    kappaS=3, psiminS=-6, mu_mut_MIC=1e-7, dilution_factor=1,
                    min_bR=2.0, min_value=-0.5, min_mic_threshold=6.25, pers_cost=0.2):
    """
    Runs the tau-leaping simulation with fixed persister levels (no persister mutations).
    Returns:
    - subpopulations, daily_most_common_mic, daily_average_mic, population_extinct_flag
    """
    # Constants
    psimaxS = bS - dS
    np.random.seed()

    # Initialize subpopulations
    subpopulations = []
    next_subpop_id = 1  # start IDs from 1

    initial_subpop = {
        'id': 0,  # initial sensitive bacteria
        'pop': int(initial_pop_size),
        'mic': S_MIC,
        'pers_level': initial_pers_level,  # Set initial persistence level (fixed)
        'bR': mic_to_br(S_MIC, min_bR=min_bR),
        'pop_PR': 0,  # Population of persisters
        'final_total_pop': 0,
    }

    subpopulations.append(initial_subpop)

    t = t0
    treatment_active = False
    current_interval_index = 0
    n_intervals = len(intervals)

    # For extinction tracking
    population_extinct = False

    # Initialize daily tracking
    daily_most_common_mic = []
    daily_average_mic = []
    next_day_time = t0 + 24  # Record at the end of each day

    while t <= tfin:
        # Calculate total population
        total_population = sum([subpop['pop'] + subpop.get('pop_PR', 0) for subpop in subpopulations])

        if total_population <= 0:
            # Population went extinct
            population_extinct = True
            break

        # Handle treatment intervals
        while current_interval_index < n_intervals and t >= intervals[current_interval_index][1]:
            current_interval_index += 1
        if current_interval_index < n_intervals:
            current_interval = intervals[current_interval_index]
            c_current = current_interval[2]
        else:
            c_current = 0

        if c_current > 0 and not treatment_active:
            treatment_active = True
            # Switch to persister state before antibiotic application
            for subpop in subpopulations:
                num_persisters = np.random.binomial(subpop['pop'], subpop['pers_level'])
                subpop['pop_PR'] = num_persisters
                subpop['pop'] -= num_persisters
                subpop['pop'] = int(subpop['pop'] * dilution_factor)
                subpop['pop_PR'] = int(subpop['pop_PR'] * dilution_factor)
        elif c_current == 0 and treatment_active:
            treatment_active = False
            # Revert persisters back to their original subpopulations
            for subpop in subpopulations:
                subpop['pop'] += subpop.get('pop_PR', 0)
                subpop['pop_PR'] = 0

        if current_interval_index < n_intervals:
            period_end = intervals[current_interval_index][1]
        else:
            period_end = tfin

        tau = min(tau_max, period_end - t, tfin - t)
        if tau <= 0:
            tau = 1e-6

        new_subpopulations = []
        updated_subpopulations = []

        for subpop in subpopulations:
            # Calculate rates for subpopulation
            mic = subpop['mic']
            bR = subpop['bR']
            pers_level = subpop['pers_level']
            pop = subpop['pop']
            pop_PR = subpop.get('pop_PR', 0)

            if c_current == 0:
                aR = 0
                aPR = 0
            else:
                psiminR = psiminS
                psimaxR = bR - dS
                aR = (psimaxR - psiminR) * (c_current / mic) ** kappaS / (
                    (c_current / mic) ** kappaS - psiminR / psimaxR
                )
                psimaxPR = 0.05
                psiminPR = mic_mapping(mic, min_value=min_value, min_mic_threshold=min_mic_threshold)
                aPR = (psimaxPR - psiminPR) * (c_current / mic) ** kappaS / (
                    (c_current / mic) ** kappaS - psiminPR / psimaxPR
                )

            # Net growth rate for subpopulation
            birth_rate_per_capita = max(
                0, (bR - cS * total_population / K) * (1 - pers_cost * pers_level)
            )
            death_rate_per_capita = max(0, dS + aR)

            # Rates for subpop['pop']
            birth_rate = max(0, birth_rate_per_capita * pop)
            death_rate = max(0, death_rate_per_capita * pop)

            # Mutation in MIC only
            mutations_MIC = poisson.rvs(max(0, birth_rate * mu_mut_MIC * tau))

            # Rates for subpop['pop_PR']
            birth_PR_rate_per_capita = max(0, bP - cS * total_population / K)
            death_PR_rate_per_capita = max(0, dP + aPR)
            birth_PR_rate = max(0, birth_PR_rate_per_capita * pop_PR)
            death_PR_rate = max(0, death_PR_rate_per_capita * pop_PR)

            # Simulate events for subpop['pop']
            births = poisson.rvs(max(0, birth_rate * tau))
            deaths = poisson.rvs(max(0, death_rate * tau))

            # Simulate events for subpop['pop_PR']
            births_PR = poisson.rvs(max(0, birth_PR_rate * tau))
            deaths_PR = poisson.rvs(max(0, death_PR_rate * tau))

            # Update subpop['pop']
            subpop['pop'] += births - deaths - mutations_MIC
            subpop['pop'] = max(subpop['pop'], 0)

            # Update subpop['pop_PR']
            subpop['pop_PR'] = pop_PR + births_PR - deaths_PR
            subpop['pop_PR'] = max(subpop['pop_PR'], 0)

            # Collect mutations in MIC
            for _ in range(mutations_MIC):
                delta_MIC = lognorm(s=sigma, scale=np.exp(mu)).rvs()
                new_mic = subpop['mic'] + delta_MIC
                new_bR = mic_to_br(new_mic, min_bR=min_bR)
                new_subpop = {
                    'id': next_subpop_id,
                    'pop': 1,
                    'mic': new_mic,
                    'pers_level': subpop['pers_level'],
                    'bR': new_bR,
                    'pop_PR': 0,
                    'final_total_pop': 0,
                }
                next_subpop_id += 1
                new_subpopulations.append(new_subpop)

            # Keep subpop if it still has population
            if subpop['pop'] > 0 or subpop.get('pop_PR', 0) > 0:
                updated_subpopulations.append(subpop)

        # Update subpopulations
        subpopulations = updated_subpopulations + new_subpopulations

        # Update time
        t += tau

        # Record daily MIC data
        while t >= next_day_time:
            # Collect mic_population_pairs
            mic_population_pairs = []
            for s in subpopulations:
                total_pop = s['pop'] + s.get('pop_PR', 0)
                mic_population_pairs.append((s['mic'], total_pop))

            # Find the mic with the largest population
            if mic_population_pairs:
                most_common_mic_day, _ = max(mic_population_pairs, key=lambda x: x[1])
            else:
                most_common_mic_day = S_MIC

            # Calculate average MIC
            total_pop = sum([pop for _, pop in mic_population_pairs])
            if total_pop > 0:
                total_mic_weighted = sum([mic * pop for mic, pop in mic_population_pairs])
                average_mic_day = total_mic_weighted / total_pop
            else:
                average_mic_day = None

            daily_most_common_mic.append(most_common_mic_day)
            daily_average_mic.append(average_mic_day)

            next_day_time += 24

    return (subpopulations, daily_most_common_mic, daily_average_mic, population_extinct)

def run_simulation_with_mic(params):
    mu_mut_MIC = params['mu_mut_MIC']
    kappaS = params['kappaS']
    psiminS = params['psiminS']
    dilution_factor = params['dilution_factor']
    min_bR = params['min_bR']
    min_value = params['min_value']
    sigma = params['sigma']
    initial_MIC = 2  # Starting MIC for the wild-type
    initial_pers_level = params['initial_pers_level']  # Starting persistence level (fixed)
    tfin = params['tfin']
    c = params['c']
    # We use 'c' for min_mic_threshold as per original script
    min_mic_threshold = c  
    simulation_id = params['simulation_id']
    treatment_duration = params['treatment_duration']
    growth_duration = params['growth_duration']
    total_time = tfin
    pers_cost = params['pers_cost']
    initial_pop_size = params['initial_pop_size']
    K = params['K']

    # Generate antibiotic intervals
    antibiotic_intervals = generate_intervals(
        growth_duration=growth_duration,
        treatment_duration=treatment_duration,
        total_time=total_time,
        treatment_concentration=c
    )

    # Calculate number of cycles
    cycle_duration = growth_duration + treatment_duration
    number_of_cycles = int(np.floor(total_time / cycle_duration))

    (subpopulations, daily_most_common_mic,
     daily_average_mic, population_extinct) = RUN_tau_leaping(
         tfin, initial_MIC, antibiotic_intervals, initial_pers_level=initial_pers_level,
         mu=mu, sigma=sigma, K=K, initial_pop_size=initial_pop_size,
         kappaS=kappaS, psiminS=psiminS, mu_mut_MIC=mu_mut_MIC,
         dilution_factor=dilution_factor, min_bR=min_bR, min_value=min_value,
         min_mic_threshold=min_mic_threshold, pers_cost=pers_cost
    )

    # At the end, collect MIC-population pairs
    mic_population_pairs = []
    for subpop in subpopulations:
        total_pop = subpop['pop'] + subpop.get('pop_PR', 0)
        mic_population_pairs.append((subpop['mic'], total_pop))

    # Most common MIC in final population
    if mic_population_pairs:
        most_common_mic, _ = max(mic_population_pairs, key=lambda x: x[1])
    else:
        most_common_mic = initial_MIC

    # Extinction status
    extinct = 1 if population_extinct else 0

    # Prepare result dictionary
    result = {
        'simulation_id': simulation_id,
        'tfin': tfin,
        'c': c,
        'initial_pers_level': initial_pers_level,
        'sigma': sigma,
        'mu_mut_MIC': mu_mut_MIC,
        'kappaS': kappaS,
        'psiminS': psiminS,
        'dilution_factor': dilution_factor,
        'min_bR': min_bR,
        'min_value': min_value,
        'treatment_duration': treatment_duration,
        'growth_duration': growth_duration,
        'most_common_mic': most_common_mic,
        'pers_cost': pers_cost,
        'initial_pop_size': initial_pop_size,
        'K': K,
        'number_of_cycles': number_of_cycles,
        'extinct': extinct
    }

    # Daily MIC data
    n_days = int(np.ceil(tfin / 24))
    if len(daily_most_common_mic) < n_days:
        daily_most_common_mic.extend([None] * (n_days - len(daily_most_common_mic)))
    if len(daily_average_mic) < n_days:
        daily_average_mic.extend([None] * (n_days - len(daily_average_mic)))

    for day in range(1, n_days + 1):
        mic_key = f'most_common_mic_day_{day}'
        avg_mic_key = f'average_mic_day_{day}'
        result[mic_key] = daily_most_common_mic[day - 1]
        result[avg_mic_key] = daily_average_mic[day - 1]

    return result


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a dense list of c values: 24 steps from 2 to 25
    c_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 50, 100]

    # Create a dense list of growth_duration values: 24 steps from 48 down to 1
    growth_durations = [48, 40, 36, 28, 24, 20, 16, 12, 10, 8, 6, 4, 2]

    # Fixed treatment duration
    treatment_duration = 5

    # Total simulation time (e.g., 15 days)
    tfin = 24 * 12

    # Ensure output directory exists
    results_dir = 'results_pop_size_plot'
    if rank == 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    comm.Barrier()

    simulation_id = rank
    all_results = []

    # Loop over all pairs of (c, growth_duration)
    for c, growth_duration in zip(c_values, growth_durations):
        # Other parameter sets as before
        sigma_values = [0.7]
        initial_pers_levels = [0.8, 5e-5]
        mu_mut_MIC_values = [1e-6]
        kappaS_values = [2]
        psiminS_values = [-4]
        dilution_factor_values = [1]
        min_bR_values = [2]
        min_value_values = [-0.5]
        pers_cost_values = [0]
        initial_pop_sizes = [1e6, 1e7, 1e8, 1e9]
        repeats = 500

        all_params = []

        # Generate all parameter combinations
        for initial_pop_size in initial_pop_sizes:
            K = initial_pop_size
            for sigma_value in sigma_values:
                for mu_mut_MIC in mu_mut_MIC_values:
                    for kappaS in kappaS_values:
                        for psiminS in psiminS_values:
                            for dilution_factor in dilution_factor_values:
                                for min_bR in min_bR_values:
                                    for min_value in min_value_values:
                                        for pers_cost in pers_cost_values:
                                            for initial_pers_level in initial_pers_levels:
                                                for repeat in range(repeats):
                                                    params = {
                                                        'simulation_id': simulation_id,
                                                        'tfin': tfin,
                                                        'c': c,
                                                        'initial_pers_level': initial_pers_level,
                                                        'kappaS': kappaS,
                                                        'psiminS': psiminS,
                                                        'mu_mut_MIC': mu_mut_MIC,
                                                        'dilution_factor': dilution_factor,
                                                        'min_bR': min_bR,
                                                        'min_value': min_value,
                                                        'sigma': sigma_value,
                                                        'treatment_duration': treatment_duration,
                                                        'growth_duration': growth_duration,
                                                        'pers_cost': pers_cost,
                                                        'initial_pop_size': int(initial_pop_size),
                                                        'K': int(K),
                                                    }
                                                    simulation_id += size
                                                    all_params.append(params)

        # Distribute tasks among MPI processes
        params_per_rank = [all_params[i::size] for i in range(size)]
        my_params = params_per_rank[rank]

        for idx, params in enumerate(my_params):
            result = run_simulation_with_mic(params)
            all_results.append(result)
            current_simulation = idx + 1
            total_my_simulations = len(my_params)
            print(f"Rank {rank}: Simulation {current_simulation}/{total_my_simulations} completed.")

        print(f'Rank {rank}: Done for c={c}, growth_duration={growth_duration}')

    # Save results
    results_csv_file = os.path.join(results_dir, f'simulation_results_rank_{rank}.csv')
    n_days = int(np.ceil(tfin / 24))
    fieldnames = [
        'simulation_id',
        'tfin', 'c', 'initial_pers_level', 'sigma', 'mu_mut_MIC', 'kappaS',
        'psiminS', 'dilution_factor', 'min_bR', 'min_value', 'pers_cost',
        'treatment_duration', 'growth_duration', 'most_common_mic',
        'initial_pop_size', 'K', 'number_of_cycles', 'extinct'
    ]
    # Add daily MIC columns
    fieldnames.extend([f'most_common_mic_day_{day}' for day in range(1, n_days + 1)])
    fieldnames.extend([f'average_mic_day_{day}' for day in range(1, n_days + 1)])

    with open(results_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Ensure daily keys are present
            for day in range(1, n_days + 1):
                mic_key = f'most_common_mic_day_{day}'
                avg_mic_key = f'average_mic_day_{day}'
                result.setdefault(mic_key, None)
                result.setdefault(avg_mic_key, None)
            writer.writerow(result)

    comm.Barrier()
    if rank == 0:
        with open(os.path.join(results_dir, 'done'), 'a') as f:
            f.write(
                'Simulation completed for dense c values (2→25) and growth_durations (48→1) '
                'with fixed treatment duration=5.\n'
            )


if __name__ == "__main__":
    main()
