import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_reward_history(csv_path, save_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 6))
    
    # 1. Raw Reward Plot (with low alpha for context)
    plt.plot(df['Episode'], df['Reward'], color='blue', alpha=0.1, label='Raw Reward')
    
    # 2. Moving Average (Smooth curve)
    window = 50
    if len(df) > window:
        df['MA'] = df['Reward'].rolling(window=window).mean()
        plt.plot(df['Episode'], df['MA'], color='red', linewidth=2, label=f'Moving Average (n={window})')

    plt.title('Reward History: Q-Learning in Dynamic Environment', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_reward_history("assets/reward_history.csv", "assets/reward_over_time.png")
