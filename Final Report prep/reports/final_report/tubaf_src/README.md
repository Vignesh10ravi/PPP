# TUBAF LaTeX

This package provides document classes and style files for the corporate design (CD) of the Technical University Bergakademie Freiberg (TUBAF).
The document classes are based on the excellent `koma-script` classes.
There is also a class for making presentations based on `beamer`.

## Provided Functionality

The following classes are provided:

- `tubaf-article` for smaller article-like documents with sections as uppermost sectioning (based on `scrartcl`, serves equivalent purposes as standard `article`)
- `tubaf-protocol` for protocols in student's experimental courses, especially for chemistry or other sciences (based on `tubaf-article`)
- `tubaf-report` for larger report-like documents with chapters as uppermost sectioning (based on `scrreprt`, serves the same purposes as standard `report`)
- `tubaf-thesis` for theses (based on `tubaf-report`)
- `tubaf-script` for lecture notes and scripts (based on `tubaf-report`)
- `tubaf-letter` for letters (based on `scrlttr2`)
- `tubaf-beamer` for presentation slides (based on `beamer`)

The following style files are provided:

- `tubaf-color` defining the CD color palette using `xcolor`
- `tubaf-logo` defining university logo and signet drawn in `tikz`
- `tubaf-silhouette` defining university silhouette in `tikz`
- `tubaf-icons` defining icon graphics in `tikz`
- `tubaf-trapezoid` defining trapezoid style element in `tikz`
- `tubaf-fonts` defining font settings
- `tubaf-title` settings for title and title page
- `tubaf-marks` settings for marks in header and footer

The classes and styles support only a limited set of options and variants.
For faculty or institute specifics, there is a [catalogue of code snippets](https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/19727876096/CourseNode/1701746998961868011/wiki/Index) (only accessible by university members) to be pasted into the document.

## Contact

For help and further information, please contact us via the following ways:

- Mail at `latex@tu-freiberg.de`
- The forum at [OPAL](https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/19727876096) (only for university members)
- The issue tracker at [GitLab](https://gitlab.hrz.tu-chemnitz.de/tubaf-latex/tubaf-latex/-/issues) (only for university members)

The source code is hosted on the [GitLab instance of TU Chemnitz](https://gitlab.hrz.tu-chemnitz.de/tubaf-latex/tubaf-latex) (only accessible for members of TUBAF or other Saxon universities).

## Installation

Instructions can be found in the main documentation file (`tubaf.md`/`tubaf.pdf`) in German as well English language.

## Building the Package

> These steps are only needed to *build* the package, which is mainly aimed at developers. If you want to install the package manually, please use one of the provided release archives from CTAN or [OPAL](https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/19727876096/CourseNode/1710386968343049003).

### Option 1: Docker Builder (Recommended)

The easiest way to build the package is using the provided Docker container with the Makefile:

```bash
# Build the Docker image once
make build-image

# Build the project
make build-project

# Create CTAN package
make build-ctan

# Or build everything at once
make build-all
```

#### Available Makefile Targets:

```bash
make build-image    # Build Docker image
make build-project  # Build LaTeX project + documentation
make build-ctan     # Create CTAN package
make build-all      # Build project + CTAN package
make clean-project  # Clean build artifacts
make clean-image    # Remove Docker image
make clean-docker   # Clean Docker system (images, containers, volumes)
make clean-all      # Clean everything (project + Docker)
make shell          # Start interactive shell
make rebuild        # Rebuild image (clean + build)
```

#### Manual Docker Commands (if needed):

<details>
<summary>Click to expand manual Docker commands</summary>

```bash
# Build the Docker image
docker build -t tubaf-latex-builder .

# Build the project
docker run --rm -v $(pwd):/workspace -v /workspace/.venv tubaf-latex-builder build.sh

# Create CTAN package
docker run --rm -v $(pwd):/workspace -v /workspace/.venv tubaf-latex-builder ctan.sh

# Clean build artifacts
docker run --rm -v $(pwd):/workspace -v /workspace/.venv tubaf-latex-builder clean.sh
```

</details>

For more Docker usage examples, see [DOCKER.md](DOCKER.md).

### Option 2: Local Build

To build the package locally, first create a local Python virtual environment using [`uv`](https://docs.astral.sh/uv)

```
uv sync
```

Then, run \[`pytask`\] within that environment to execute the full build procedure.

```
uv run pytask
```

The final package will be places in `dist/`.

### Common Issues

- **Build Failures**: The complete TeXLive installation should handle most LaTeX compilation issues. Use `make shell` for interactive debugging.
- **Graphics Library**: On some systems the graphics library `cairo` (only Linux) is not installed when building locally. `cairo` can not be automatically installed by `pip` and must be installed via the systems package manager.
  ```
  [pipenv.exceptions.InstallError]:             Package cairo was not found in the pkg-config search path.
  [pipenv.exceptions.InstallError]:             Perhaps you should add the directory containing `cairo.pc'
  [pipenv.exceptions.InstallError]:             to the PKG_CONFIG_PATH environment variable
  [pipenv.exceptions.InstallError]:             No package 'cairo' found
  ```

## License

The package is distributed under the terms of the [MIT LICENSE](LICENSE).
Note that, although the package is licensed under MIT terms, the TUBAF signet, logos and other corporate design elements are protected as a trademark and their use is only permitted to university members.
Unauthorized use, pretending to be a member of the university or to speak in behalf of the university will be prosecuted.
However, you are free to use, modify and redistribute the package or parts of it as long as you do not violate the trademark rights.
