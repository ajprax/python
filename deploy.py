from glob import glob
from os import remove
from os.path import dirname, join
from subprocess import check_call

if __name__ == "__main__":
    for path in glob(join(dirname(__file__), "dist", "*")):
        remove(path)

    check_call("python -m build".split())
    check_call("python -m twine upload --repository pypi dist/* --verbose".split())
