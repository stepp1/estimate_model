import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from absl import app, flags, logging
from matplotlib import pyplot as plt

import bm.estimate
from bm.experiments import configs, dataset_utils, plotting_utils, training

flags.DEFINE_string("dataset", "blobs", "Dataset to use")
flags.DEFINE_integer("seed", 1234, "Random seed")
flags.DEFINE_integer("n_samples", 500_000, "Number of samples")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_float("test_size", 0.2, "Test size")
flags.DEFINE_boolean("transparent", False, "Transparent background")
flags.DEFINE_boolean("skip_joint", False, "Skip joint distribution")
flags.DEFINE_integer(
    "top_k_features",
    None,
    "Top k features to use for joint distribution",
)

FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)


def run_experiment_1(
    dataset_name: str,
    hp_configs: list[dict[str, Any]],
    seed: int = 1234,
    transparent: bool | None = None,
) -> list[dict[str, Any]]:
    """Executes Experiment 1.

    Hyperparameter Sweep and Distribution Analysis.
    """
    print(f"\n--- Running Experiment 1 for {dataset_name} ---")
    experiment_results = []

    f_with_loss, axs_with_loss = plt.subplots(
        nrows=2,
        ncols=len(hp_configs),
        figsize=(40, 8),
    )
    f_without_loss, axs_without_loss = plt.subplots(
        nrows=1,
        ncols=len(hp_configs),
        figsize=(40, 4),
    )

    analytical_losses = []
    for i, ((config_name, hp_config), ax_loss, ax_wo_loss) in enumerate(
        zip(hp_configs, axs_with_loss[0], axs_without_loss, strict=True),
    ):
        print(f"  Generating data with HP config: {config_name}")

        dataset = dataset_utils.prepare_dataset(
            dataset_name,
            hp_config,
            seed=seed,
        )

        plotting_utils.plot_dataset_distribution(
            dataset.x,
            dataset.y,
            dataset_name,
            (config_name, hp_config),
            i,
            top_k_features=FLAGS.top_k_features,
            skip_joint=FLAGS.skip_joint,
        )
        joint_metadata, marginal_metadata, analytical_loss = (
            bm.estimate.estimate_synthetic_mi(
                dataset.x,
                dataset.y,
                skip_joint=FLAGS.skip_joint,
                top_k_features=FLAGS.top_k_features,
            )
        )
        analytical_losses.append(
            analytical_loss["joint"]
            if not FLAGS.skip_joint
            else analytical_loss["marginal"]
        )
        ax_loss.scatter(
            dataset.x[:, 0],
            dataset.x[:, 1],
            c=dataset.y,
            cmap="turbo",
            s=1,
            alpha=0.7,
        )
        ax_loss.set_title(f"{config_name.replace('_', ' ').title()}")

        ax_wo_loss.scatter(
            dataset.x[:, 0],
            dataset.x[:, 1],
            c=dataset.y,
            cmap="turbo",
            s=1,
            alpha=0.7,
        )
        ax_wo_loss.set_title(
            f"{config_name.replace('_', ' ').title()}\nAnalytical Loss: {analytical_loss['joint']:.3f} nats",
        )
        experiment_results.append(
            {
                "dataset": dataset_name,
                "hp_config": (config_name, hp_config),
                "n_samples": dataset.x.shape[0],
                "input_dim": dataset.x.shape[1],
                "n_classes": len(np.unique(dataset.y)),
            },
        )

    # Reshape analytical losses to 2D array (1xN configuration)
    analytical_losses = np.array(analytical_losses).reshape(1, -1)
    vmin = analytical_losses.min()
    vmax = analytical_losses.max()

    for i, ((config_name, hp_config), ax_loss) in enumerate(
        zip(hp_configs, axs_with_loss[1], strict=True),
    ):
        # Plot single value as 1x1 heatmap tile
        mesh = ax_loss.pcolormesh(
            analytical_losses[:, i].reshape(1, 1),  # Reshape to 2D
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_loss.text(
            0.5,
            0.5,
            f"{analytical_losses[:, i][0]:.3f}",
            ha="center",
            va="center",
            color="white"
            if analytical_losses[:, i][0] > (vmin + vmax) / 2
            else "black",
            fontsize=16,
            bbox={
                "facecolor": "black"
                if analytical_losses[:, i][0] > (vmin + vmax) / 2
                else "white",
                "alpha": 0.7,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.3",
            },
        )
        ax_loss.set_xticks([])
        ax_loss.set_yticks([])

    # Add colorbar below the figure
    f_with_loss.subplots_adjust(bottom=0.15)  # Make space at bottom
    cax = f_with_loss.add_axes([0.36, 0.08, 0.3, 0.03])  # [left, bottom, width, height]
    cb = f_with_loss.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        label="Analytical Loss (nats)",
    )
    cb.outline.set_edgecolor("none")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=10, width=0.5, length=2)
    cb.ax.set_facecolor("#f0f0f0")  # Light background for contrast
    cb.set_label(
        "Analytical Loss (nats)",
        fontsize=12,
        weight="bold",
        labelpad=8,
    )
    cb.ax.set_facecolor("#f0f0f0")  # Light background for contrast

    # Remove tight_layout if conflicting
    f_with_loss.savefig(
        f"experiment1_plot_{dataset_name}.png",
        bbox_inches="tight",
        dpi=300,
        transparent=transparent,
    )
    plt.close(f_with_loss)

    f_without_loss.tight_layout()
    f_without_loss.savefig(
        f"experiment1_plot_{dataset_name}_without_loss.png",
        bbox_inches="tight",
        dpi=300,
        transparent=transparent,
    )
    plt.close(f_without_loss)
    print(f"--- Experiment 1 for {dataset_name} complete. ---")
    return experiment_results


def run_experiment_2(
    dataset_name: str,
    selected_hp_configs: list[dict[str, Any]],
    seed: int = 1234,
    batch_size: int = 256,
    test_size: float = 0.2,
    transparent: bool | None = None,
) -> list[dict[str, Any]]:
    """Executes Experiment 2.

    Synthetic Distribution Fitting, PyTorch MLP Training, and Loss Analysis.
    """
    print(f"\n--- Running Experiment 2 for {dataset_name} (PyTorch) ---")
    all_results = []
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
    )
    print(f"  Using device: {device}")

    for i, (config_name, hp_config) in enumerate(selected_hp_configs):
        print(
            f"  Processing HP config {i + 1}/{len(selected_hp_configs)}: {config_name}",
        )
        dataset = dataset_utils.prepare_dataset(
            dataset_name,
            hp_config,
            batch_size=batch_size,
            test_size=test_size,
            device=device,
            seed=seed,
        )

        # 2. Fit Synthetic Distribution and Calculate Metrics using your framework
        print("Fitting synthetic distribution and calculating metrics...")
        joint_metadata, marginal_metadata, analytical_loss = (
            bm.estimate.estimate_synthetic_mi(
                dataset.x,
                dataset.y,
                skip_joint=FLAGS.skip_joint,
                top_k_features=FLAGS.top_k_features,
            )
        )
        metadata = dataset.metadata
        metadata.update(
            {
                "config_name": config_name,
                "hp_config": hp_config,
                "mi": joint_metadata["mi"]["joint"],
                "entropy": joint_metadata["symbol_entropy"],
                "analytical_loss": analytical_loss["joint"],
            },
        )

        # 3. PyTorch MLP Training
        print("Training MLP model...")
        mlp_history = training.train_mlp_model(
            dataset.train_loader,
            dataset.test_loader,
            num_epochs=20,
            metadata=metadata,
            device=device,
        )
        metadata.update(
            {
                "mlp_history": mlp_history,
            },
        )
        joint_analytical_loss = (
            analytical_loss["joint"]
            if not FLAGS.skip_joint
            else analytical_loss["marginal"]
        )
        marginal_analytical_loss = analytical_loss["marginal"]
        print(
            f"  MLP training complete. "
            f"Final test loss: {mlp_history['final_test_loss']:.3f} nats, "
            f"Expected analytical loss: {joint_analytical_loss:.3f} nats "
            f"(marginal: {marginal_analytical_loss:.3f} nats), "
            f"MI: {metadata['mi']:.3f} nats, "
            f"Entropy: {metadata['entropy']:.3f} nats, "
            f"Final test accuracy: {mlp_history['final_test_accuracy']:.3f}",
        )
        plotting_utils.plot_history(
            mlp_history,
            joint_analytical_loss,
            dataset_name,
            config_name,
            output_filename=f"experiment2_history_{i}_{dataset_name}_{config_name}.png",
            transparent=transparent,
        )
        all_results.append(metadata)

    output_filename = f"experiment2_results_pytorch_{dataset_name}.json"
    print(f"Saving results to {output_filename}")
    with Path(output_filename).open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {output_filename}")

    plotting_utils.plot_all_results(
        all_results,
        dataset_name,
        [config_name for config_name, _ in selected_hp_configs],
        output_filename=f"experiment2_all_results_pytorch_{dataset_name}.png",
        transparent=transparent,
    )

    plotting_utils.plot_loss_comparison(
        all_results,
        dataset_name,
        [config_name for config_name, _ in selected_hp_configs],
        output_filename=f"experiment2_loss_comparison_pytorch_{dataset_name}.png",
        transparent=transparent,
    )
    print(f"--- Experiment 2 for {dataset_name} complete. ---")
    return all_results


def run_experiment_3(
    dataset_name: str,
    hp_start: dict[str, Any],
    hp_end: dict[str, Any],
    num_interpolation_steps: int = 5,
    seed: int = 1234,
    batch_size: int = 256,
    test_size: float = 0.2,
) -> list[dict[str, Any]]:
    """Executes Experiment 3: Loss Interpolation between Distributions."""

    print(f"\n--- Running Experiment 3 for {dataset_name}: Interpolation (PyTorch) ---")
    interpolation_results = []
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
    )
    print(f"  Using device: {device}")

    # Generate interpolated HP configurations
    interpolated_hps = []
    interpolatable_keys = [
        k for k in hp_start if isinstance(hp_start[k], (int, float)) and k in hp_end
    ]

    for step_alpha in np.linspace(0, 1, num_interpolation_steps):
        current_hp = hp_start.copy()
        for key in interpolatable_keys:
            current_hp[key] = (
                hp_start[key] * (1 - step_alpha) + hp_end[key] * step_alpha
            )

        # Ensure integer parameters remain integers
        for k in [
            "n_samples",
            "n_features",
            "n_classes",
            "n_informative",
            "n_redundant",
            "centers",
        ]:
            if k in current_hp and isinstance(current_hp[k], float):
                current_hp[k] = int(round(current_hp[k]))

        interpolated_hps.append(current_hp)

        for i, hp_config in enumerate(interpolated_hps):
            print(
                f"  Interpolation Step {i + 1}/{num_interpolation_steps}: {hp_config}",
            )

            dataset = dataset_utils.prepare_dataset(
                dataset_name,
                hp_config,
                device=device,
                batch_size=batch_size,
                test_size=test_size,
                seed=seed,
            )

            # 2. Fit Synthetic Distribution and Calculate Metrics using your framework
            print("    Fitting synthetic distribution and calculating metrics...")
            joint_metadata, marginal_metadata, analytical_loss = (
                bm.estimate.estimate_synthetic_mi(
                    dataset.x,
                    dataset.y,
                    grid_size=10,
                    skip_joint=FLAGS.skip_joint,
                    top_k_features=FLAGS.top_k_features,
                )
            )
            joint_analytical_loss = (
                analytical_loss["joint"]
                if not FLAGS.skip_joint
                else analytical_loss["marginal"]
            )
            # 3. PyTorch MLP Training
            mlp_history = training.train_mlp_model(
                dataset.train_loader,
                dataset.test_loader,
                metadata=dataset.metadata,
                device=device,
            )

            interpolation_results.append(
                {
                    "step": i,
                    "alpha": step_alpha,  # Use step_alpha for consistent plotting
                    "hp_config": hp_config,
                    "analytical_loss": joint_analytical_loss,
                    "mlp_history": mlp_history,
                    "n_samples": dataset.x.shape[0],
                    "input_dim": dataset.x.shape[1],
                    "n_classes": len(np.unique(dataset.y)),
                },
            )

    output_filename = f"experiment3_results_pytorch_{dataset_name}_interpolation.json"
    with Path(output_filename).open("w", encoding="utf-8") as f:
        json.dump(interpolation_results, f, indent=4)
    print(f"Results saved to {output_filename}")

    plotting_utils.plot_loss_interpolation(
        interpolation_results,
        dataset_name,
        output_filename=f"experiment3_loss_interpolation_pytorch_{dataset_name}.png",
    )
    print(f"--- Experiment 3 for {dataset_name} complete. ---")
    return interpolation_results


def main(argv: list[str]):
    del argv
    # Experiment 1 HPs
    base_hps = configs.get_config(FLAGS.dataset, FLAGS.n_samples, FLAGS.seed)

    # Run the experiments for selected datasets
    print("\n=========== Running All Experiments ===========")

    print("\n--- Running Experiment 1 ---")
    run_experiment_1(
        FLAGS.dataset,
        base_hps,
        seed=FLAGS.seed,
        transparent=FLAGS.transparent,
    )
    # run_experiment_1("make_moons", make_moons_s1_hps)
    # run_experiment_1("make_classification", make_classification_s1_hps)
    # run_experiment_1("load_iris", load_iris_s1_hps)

    print("\n--- Running Experiment 2 ---")
    run_experiment_2(
        FLAGS.dataset,
        base_hps,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        test_size=FLAGS.test_size,
        transparent=FLAGS.transparent,
    )
    # run_experiment_2("make_moons", make_moons_s2_hps)

    # print("\n--- Running Experiment 3 ---")
    # run_experiment_3(
    #     "make_blobs",
    #     hp_start_blobs,
    #     hp_end_blobs,
    #     num_interpolation_steps=7,
    #     seed=FLAGS.seed,
    #     batch_size=FLAGS.batch_size,
    #     test_size=FLAGS.test_size,
    # )

    print("\n=========== All Experiment Experiments Completed ===========")


# --- Example Usage (main execution block) ---
if __name__ == "__main__":
    app.run(main)
