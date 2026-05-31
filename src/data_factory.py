# data_factory.py

def get_loaders(dataset: str, **kwargs):
    """
    Returns: (src_train, src_test, tgt_train, tgt_test, val_loader)
    Each loader yields (image_tensor, label, domain_label).
    domain_label: 0 = source, 1 = target.
    """
    if dataset == "colored_mnist":
        from data_colored_mnist import get_colored_mnist_loaders
        # only pass kwargs this loader understands
        return get_colored_mnist_loaders(
            root=kwargs["data_root"],
            batch_size=kwargs["batch_size"],
            source_color_prob=kwargs.get("source_color_prob", 0.99),
            target_color_prob=kwargs.get("target_color_prob", 0.10),
            seed=kwargs.get("seed", 42),
            num_workers=kwargs.get("num_workers", 2),
            binary_labels=kwargs.get("binary_labels", False),
        )

    elif dataset == "rotated_mnist":
        from data_rotated_mnist import get_rotated_mnist_loaders
        return get_rotated_mnist_loaders(
            root=kwargs["data_root"],
            batch_size=kwargs["batch_size"],
            source_angle=kwargs.get("source_angle", 0),
            target_angle=kwargs.get("target_angle", 45),
            seed=kwargs.get("seed", 42),
            num_workers=kwargs.get("num_workers", 2),
        )

    elif dataset == "mnist_c":
        from data_mnist_c import get_mnist_c_loaders
        return get_mnist_c_loaders(
            root=kwargs["data_root"],
            batch_size=kwargs["batch_size"],
            corruption=kwargs.get("corruption", "stripe"),
            seed=kwargs.get("seed", 42),
            num_workers=kwargs.get("num_workers", 2),
        )

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Choose: colored_mnist, rotated_mnist, mnist_c")
    