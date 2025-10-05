import tomllib
from glob import glob
from os import remove
from os.path import dirname, join
from subprocess import check_call


def version():
    with open("pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


if __name__ == "__main__":
    for path in glob(join(dirname(__file__), "dist", "*")):
        remove(path)

    version = version()
    check_call("python -m build".split())
    check_call("python -m twine upload --repository pypi dist/* --verbose".split())
    check_call(f"git tag {version}".split())
    check_call(f"git push origin {version}".split())
