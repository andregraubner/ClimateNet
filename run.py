import argparse
import example as example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        choices=[
            "example"
        ],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path of a model checkpoint to load",
    )

    args = parser.parse_args()
    print(vars(args))

    if args.model == "example":
        print("Running baseline example...")
        example.run(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir
        )
    else:
        raise NotImplementedError("Not implemented yet")