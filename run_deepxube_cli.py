import sys


def main() -> None:
    # Forward argv to deepxube's CLI entrypoint.
    import deepxube._cli as cli

    # Ensure argparse sees a program name.
    sys.argv = ["deepxube"] + sys.argv[1:]
    cli.main()


if __name__ == "__main__":
    main()
