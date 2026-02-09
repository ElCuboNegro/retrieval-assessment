"""{{ cookiecutter.project_name }} main entry point."""

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs):
    """Entry point for running Kedro project."""
    package_name = "{{ cookiecutter.python_package }}"
    configure_project(package_name)
    run = find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()
