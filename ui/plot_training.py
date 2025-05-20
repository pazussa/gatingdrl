import matplotlib.pyplot as plt
import os.path
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.results_plotter import plot_results
from tools.execute import execute_from_command_line
from tools.definitions import OUT_DIR


def plot_training_rewards(dirname: str):
    plot_results([dirname], None, 'timesteps', next((s for s in dirname.split(r'/') if '100' in s), None))
    filename = os.path.join(os.path.dirname(dirname), f"training_reward.png")
    print(f"saving the figure to {filename}")
    plt.savefig(filename)
    plt.show(block=False)
    plt.pause(3)  # Espera 3 segundos para mostrar la gráfica
    plt.close()


if __name__ == '__main__': 
    execute_from_command_line(plot_training_rewards)
    