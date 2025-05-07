import matplotlib.pyplot as plt
import seaborn as sns


def analyze_parameter_effects(results_file):
    # Load results
    df = pd.read_csv(results_file)

    # Set up the figure grid for multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Window Size vs Repeatability for different distortion levels
    for level in [1, 2, 3, 4, 5]:
        subset = df[df['distortion_level'] == level]
        sns.lineplot(x='window_size', y='repeatability',
                     data=subset, marker='o', label=f'Level {level}', ax=axes[0, 0])
    axes[0, 0].set_title('Window Size Effect on Repeatability')
    axes[0, 0].set_ylabel('Repeatability Score')

    # 2. K-value vs Repeatability for different distortion levels
    for level in [1, 3, 5]:
        subset = df[df['distortion_level'] == level]
        means = subset.groupby('k_value')['repeatability'].mean().reset_index()
        sns.lineplot(x='k_value', y='repeatability',
                     data=means, marker='o', label=f'Level {level}', ax=axes[0, 1])
    axes[0, 1].set_title('K-value Effect on Repeatability')

    # 3. Aperture Size vs Localization for different distortion levels
    for level in [1, 3, 5]:
        subset = df[df['distortion_level'] == level]
        sns.lineplot(x='aperture_size', y='localization_accuracy',
                     data=subset, marker='o', label=f'Level {level}', ax=axes[1, 0])
    axes[1, 0].set_title('Aperture Size Effect on Localization Accuracy')
    axes[1, 0].set_ylabel('Average Corner Distance (pixels)')

    # 4. Heatmap for best parameter combinations
    pivot = df.pivot_table(
        index='window_size',
        columns='k_value',
        values='repeatability',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Repeatability Heatmap (Window Size vs K-value)')

    plt.tight_layout()
    plt.savefig(f'{results_file.split(".")[0]}_analysis.png', dpi=300)
    plt.show()
