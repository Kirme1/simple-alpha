from .screen import build_arg_parser, run

if __name__ == "__main__":
    run(build_arg_parser().parse_args())
