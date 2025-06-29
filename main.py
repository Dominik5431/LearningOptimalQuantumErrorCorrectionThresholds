import argparse

from main_qec import main_qec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--distance", type=int, default=5, help="Distance of the QEC code.")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise strength p.")
    parser.add_argument("--noise_model", type=str, choices=["bitflip", "depolarizing", "phenomenological", "circuit-level"], default="depolarizing", help="Noise model.")
    parser.add_argument("--task", type=str, choices=["train", "decoding", "plot_ci", "plot_log_error", "attention", "loss", "plot-single-latent", "reconstruction", "raw-data", "plot-latent-reconstruction", "plot-vs-raw"], default="train", help="Task: Train network, evaluate network or plot the results.")

    args = parser.parse_args()

    if args.task not in ["train", "decoding", "plot_ci", "plot_log_error", "attention", "loss"]:
        raise ValueError(f'Task {args.task} not supported with method {args.method}.')
    task_dict = {"train": 1, "decoding": 4, "plot_ci": 100, "plot_log_error": 400, "attention": 5, "loss": 8}
    task = task_dict[args.task]
    if args.noise_model != "depolarizing" and args.task == "attention":
        raise ValueError("Attention plot only implemented for depolarizing noise.")
    main_qec(distance=args.distance, task=task, noise_model=args.noise_model, noise=args.noise)

