# Contributing

## Getting Help

Ask contributor directly.

## Development Practices and Tooling

### Development Workflow

We loosely practice [trunk-based-development](https://trunkbaseddevelopment.com/):

- We work almost exclusively on the master branch
- We commit, push, and pull often
- We automatically run code quality checks before every commit (using [pre-commit](https://pre-commit.com/))

### Dependency Management

To manage dependencies and for package distribution we use [poetry](https://python-poetry.org/docs/) (replaces pip).

## TODO: Developer Installation

There are three required steps and one optional:

1. Optional: Install miniconda and create an environment
1. Install poetry
1. Install the ourbench package using poetry
1. Activate pre-commit for the repository

For instructions see below.

### 1. Optional: Install miniconda and create an environment

To manage python versions install e.g., miniconda with

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/.conda  # Change to place of preference
rm install_miniconda.sh
```

Consider running `~/.conda/bin/conda init` or `~/.conda/bin/conda init zsh` .

Then finally create the environment and activate it

```bash
conda create -n ourbench python=3.7
conda activate ourbench
```

### 2. Install poetry

First, install poetry, e.g., via

```bash
wget https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -O get-poetry.py
python get-poetry.py
rm get-poetry.py
```

Then consider appending

```bash
export PATH="$HOME/.poetry/bin:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply running the export manually.

### 3. Install the ourbench Package Using poetry

Inside the main directory of ourbench run

```bash
poetry install
```

### 4. Activate pre-commit for the repository

With the python environment used to install the ourbench package run in the main directory of ourbench

```bash
pre-commit install
```

## Tooling Tips

### Poetry: Add dependencies

To install a dependency use

```bash
poetry add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [poetry documentation](https://python-poetry.org/).

### Poetry: Install dependencies added by others

When other contributors added dependencies to `pyproject.toml`, you can install them via

```bash
rm poetry.lock
poetry install
```

or also updating dependencies alongside via

```bash
poetry update
```

### Pre-commit: Do not run hooks

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Pylint: Ignore warnings

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # pylint: disable=ERROR_CODE
  ```

  Make sure to use the named version of the error (e.g., `unspecified-encoding`, not `W1514`).

- Remove warning in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

### Mypy: Ignore warnings

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # type: ignore[ERROR_CODE]
  ```

- If you know what you are doing, you can add the whole module to the `[[tool.mypy.overrides]]` section.
  This is useful e.g., when adding new files that are in early stage development.

### Black: Do not format code parts

```python
x = 2  # fmt: off
```

or for blocks

```python
# fmt: off
x = 2
y = x + 1
# fmt: on
```

### MkDocs

See [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/).

### Editorconfig

[Editorconfig](https://editorconfig.org/) allows to set line lengths and other display parameters automatically based on a `.editorconfig` file.
Many editors have [native support](https://editorconfig.org/#pre-installed) (e.g., PyCharm) so you do not need to do anything.
For other editors (e.g., VSCode), you need to install a [plugin](https://editorconfig.org/#download).
