
import optuna

# Update these if your filenames are different
db_path = "sqlite:////Users/elvisobondo/Downloads/stgnn_ultra_minimal_hyperopt_cpu.db"
study_name = "stgnn_ultra_minimal_hyperopt_cpu"
output_file = "optuna_trials_dump.txt"

# Load the study
study = optuna.load_study(study_name=study_name, storage=db_path)

with open(output_file, "w") as f:
    for trial in study.trials:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write(f"  Params:\n")
        for k, v in trial.params.items():
            f.write(f"    {k}: {v}\n")
        f.write(f"  State: {trial.state}\n")
        f.write("-" * 40 + "\n")

print(f"All trial information written to {output_file}")



'''

To extract the top 20 trials, run the following code:

import optuna

# Path to your Optuna database in Downloads
db_path = "sqlite:////Users/elvisobondo/Downloads/stgnn_ultra_minimal_hyperopt_cpu.db"
study_name = "stgnn_ultra_minimal_hyperopt_cpu"
output_file = "optuna_top20_trials.txt"

# Load the study
study = optuna.load_study(study_name=study_name, storage=db_path)

# Filter only completed trials and sort by value (ascending)
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
top_trials = sorted(completed_trials, key=lambda t: t.value)[:20]

with open(output_file, "w") as f:
    for i, trial in enumerate(top_trials, 1):
        f.write(f"Rank {i} (Trial {trial.number}):\n")
        f.write(f"  Value: {trial.value}\n")
        f.write(f"  Params:\n")
        for k, v in trial.params.items():
            f.write(f"    {k}: {v}\n")
        f.write(f"  State: {trial.state}\n")
        f.write("-" * 40 + "\n")

print(f"Top 20 trials written to {output_file}")

'''


'''

To extract the trial states, run the following code:

import optuna
from collections import Counter

db_path = "sqlite:////Users/elvisobondo/Downloads/stgnn_ultra_minimal_hyperopt_cpu.db"
study_name = "stgnn_ultra_minimal_hyperopt_cpu"
output_file = "optuna_trial_states.txt"

study = optuna.load_study(study_name=study_name, storage=db_path)

# Count trial states
states = [trial.state for trial in study.trials]
state_counts = Counter(states)

with open(output_file, "w") as f:
    for state, count in state_counts.items():
        f.write(f"{state}: {count}\n")

print("Trial state counts:")
for state, count in state_counts.items():
    print(f"{state}: {count}")
print(f"\nResults also written to {output_file}")

'''