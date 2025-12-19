import json
from pathlib import Path

import matplotlib.pyplot as plt


def find_latest_trainer_state(output_dir: Path) -> Path:
    """
    Find the trainer_state.json to use.

    Strategy:
      - If there's a trainer_state.json directly in output_dir, use that.
      - Otherwise, look for subdirectories named 'checkpoint-*',
        pick the one with the largest step number, and use its trainer_state.json.
    """
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        return root_state

    # look for checkpoint-* subdirs
    checkpoint_dirs = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            # try to parse the step number from the folder name
            try:
                step = int(p.name.split("-")[-1])
            except ValueError:
                continue
            checkpoint_dirs.append((step, p))

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No trainer_state.json found directly in {output_dir} "
            f"and no checkpoint-* directories with trainer_state.json."
        )

    # pick the checkpoint with the largest step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    _, latest_ckpt = checkpoint_dirs[-1]

    state_path = latest_ckpt / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {latest_ckpt}")
    return state_path


def plot_training_curves(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    state_path = find_latest_trainer_state(output_dir)

    print(f"Using trainer_state.json from: {state_path}")

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        raise ValueError("No log_history found in trainer_state.json")

    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []
    eval_f1s = []

    for entry in log_history:
        # Training loss logs
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])

        # Evaluation logs
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
            if "eval_f1" in entry:
                eval_f1s.append((entry["epoch"], entry["eval_f1"]))

    # 1) Train vs eval loss
    if train_epochs or eval_epochs:
        plt.figure()
        if train_epochs:
            plt.plot(train_epochs, train_losses, label="Train loss")
        if eval_epochs:
            plt.plot(eval_epochs, eval_losses, label="Eval loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "train_eval_loss.png")
        plt.close()

    # 2) Eval F1 over epochs
    if eval_f1s:
        epochs, f1s = zip(*eval_f1s)
        plt.figure()
        plt.plot(epochs, f1s, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("F1 (validation)")
        plt.title("Validation F1 over epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "eval_f1.png")
        plt.close()

    print(f"Saved curves to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot training curves (loss and F1) from trainer_state.json."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Training output_dir (contains checkpoint-* subdirs).",
    )

    args = parser.parse_args()
    plot_training_curves(args.output_dir)
