import json
from pathlib import Path


def load_final_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    final_epoch = data["history"][-1]

    return {
        "method": data["method"],
        "source_test_acc": final_epoch["source_test_acc"],
        "target_test_acc": final_epoch["target_test_acc"],
        "training_time_seconds": data["training_time_seconds"],
        "epochs": data["epochs"],
        "seed": data["seed"],
    }


def main():
    result_files = [
        "results/source_only_results.json",
        "results/deep_coral_results.json",
        "results/target_supervised_upper_bound_results.json",
    ]

    results = []

    for path in result_files:
        if not Path(path).exists():
            print(f"Missing file: {path}")
            continue

        results.append(load_final_result(path))

    print("\nFinal Results")
    print("=" * 80)
    print(f"{'Method':40s} {'Source Acc':>12s} {'Target Acc':>12s} {'Time (s)':>10s}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['method']:40s} "
            f"{result['source_test_acc']:12.4f} "
            f"{result['target_test_acc']:12.4f} "
            f"{result['training_time_seconds']:10.1f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()