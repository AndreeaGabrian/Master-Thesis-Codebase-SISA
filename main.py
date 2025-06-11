import json
from evaluation.evaluate_SISA import evaluate_sisa
from sisa.slice_data import run_split
from sisa.train_SISA import train_sisa
from sisa.unlearning import progressive_unlearning_and_evaluation
from utils.utils import get_path, get_transform
from torchvision import datasets

def load_config():
    config_path = "config_default.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_user_input(config):
    data_dir = input(f"Enter data directory path (from project root) [default: {config['data_dir']}]: ").strip()
    dataset_name = input(f"Enter dataset name [default: {config['dataset_name']}]: ").strip()

    if data_dir:
        config['data_dir'] = data_dir
    if dataset_name:
        config['dataset_name'] = dataset_name

    return config


def select_mode():
    print("\nSelect mode:")
    print("1. Train")
    print("2. Evaluate")
    print("3. Unlearn")
    choice = input("Enter the number of the mode: ").strip()

    if choice == '1':
        return "train"
    elif choice == '2':
        return "evaluate"
    elif choice == '3':
        return "unlearn"
    else:
        print("Invalid choice. Defaulting to 'evaluate'.")
        return "evaluate"


def get_train_params(config):
    try:
        num_shards = input(f"Enter number of shards [default: {config['num_shards']}]: ").strip()
        num_slices = input(f"Enter number of slices [default: {config['num_slices']}]: ").strip()
        batch_size = input(f"Enter bach size [default: {config['batch_size']}]: ").strip()
        epochs = input(f"Enter number of epochs per slice [default: {config['num_epochs_per_slice']}]: ").strip()
        training_strategy = input(f"SISA training strategy (union or no-union) [default: {config['training_strategy']}]: ").strip()
        distribution = input(f"Data distribution (random, slice-aware, shard-aware) [default: {config['distribution']}]: ").strip()

        if num_shards:
            config['num_shards'] = int(num_shards)
        if num_slices:
            config['num_slices'] = int(num_slices)
        if batch_size:
            config['batch_size'] = int(batch_size)
        if epochs:
            config['num_epochs_per_slice'] = int(epochs)
        if training_strategy:
            config['training_strategy'] = training_strategy
        if distribution:
            config['distribution'] = distribution

    except ValueError:
        print("Invalid input. Using default values.")

    return config


def save_config(config):
    save_path = get_path("utils/config.json")
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to '{save_path}'.")


def get_unlearning_steps():
    raw = input("Enter unlearning steps (comma-separated, e.g. 0.05,0.1) [default: 0.05, 0.10, 0.15]: ").strip()
    if not raw:
        return [0.05, 0.10, 0.15]
    try:
        return [float(s.strip()) for s in raw.split(",")]
    except ValueError:
        print("Invalid input. Using default steps.")
        return [0.05, 0.10, 0.15]


def main():
    config = load_config()
    config = get_user_input(config)

    mode = select_mode()

    if mode == "train":
        config = get_train_params(config)
        save_config(config)
        print("\nStarting splitting data:")
        run_split(config["distribution"])
        print("\nStarting training with config:")
        train_sisa(strategy=config["training_strategy"])
    elif mode == "evaluate":
        print("\nRunning evaluation...")
        save_config(config)
        evaluate_sisa((False, 0.05))
    elif mode == "unlearn":
        print("\nStarting unlearning process")
        save_config(config)
        steps = get_unlearning_steps()
        transform = get_transform()
        dataset = datasets.ImageFolder(get_path(config["data_dir"]), transform=transform)
        output_dir = config['output_dir']
        train_indices_file = get_path(f"{output_dir}/train_class_prob_indices.json")

        progressive_unlearning_and_evaluation(
            dataset=dataset,
            train_indices_filename=train_indices_file,
            eval_fn=lambda x: evaluate_sisa(x),
            steps=steps
        )


if __name__ == "__main__":
    main()
