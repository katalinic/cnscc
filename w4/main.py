import matplotlib.pyplot as plt
import numpy as np
import pickle


TRIAL_DURATION = 10

with open('w4/tuning_34.pickle', 'rb') as f:
    data = pickle.load(f)

stimulus = data['stim']
firing_rates = np.array([
    data['neuron{}'.format(i)] for i in range(1, 5)
])

# Q7.
fig, axes = plt.subplots(4, 1, figsize=(8, 8))
for i, ax in enumerate(axes):
    ax.scatter(stimulus, np.mean(firing_rates[i], axis=0))
plt.show()

# Q8.
print(TRIAL_DURATION * np.var(firing_rates, axis=1)
      / np.mean(firing_rates, axis=1))
