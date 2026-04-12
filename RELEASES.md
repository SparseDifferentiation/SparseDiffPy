# Release Procedures

This file documents the procedures for releasing new versions of
SparseDiffPy and its SparseDiffEngine submodule.
Adapted from the [CVXPY release procedures](https://github.com/cvxpy/cvxpy/blob/master/PROCEDURES.md).

## Versioning

SparseDiffEngine's version is defined in `SparseDiffEngine/CMakeLists.txt`:

```cmake
set(DIFF_ENGINE_VERSION_MAJOR X)
set(DIFF_ENGINE_VERSION_MINOR Y)
set(DIFF_ENGINE_VERSION_PATCH Z)
```

SparseDiffPy's version is defined in `pyproject.toml`:

```toml
version = "X.Y.Z"
```

Both repos keep their versions in sync.

## Minor Release (e.g., 0.2.0)

A minor release increments the MINOR version and resets PATCH to 0.
Always release SparseDiffEngine first, then SparseDiffPy.

### SparseDiffEngine

1. From `main`, create a release branch `release/X.Y.x`:
   ```bash
   git checkout main
   git checkout -b release/X.Y.x
   ```

2. Set the version to `X.Y.0` in `CMakeLists.txt`, commit, and tag:
   ```bash
   # Edit CMakeLists.txt: MINOR = Y, PATCH = 0
   git add CMakeLists.txt
   git commit -m "Set version to X.Y.0 for release"
   git tag vX.Y.0
   ```

3. Lay groundwork for the next patch release on this branch.
   Bump PATCH to 1 and commit. *Do not* tag this commit:
   ```bash
   # Edit CMakeLists.txt: PATCH = 1
   git add CMakeLists.txt
   git commit -m "Begin X.Y.1 development"
   ```

4. Switch to `main` and bump to the next minor version for development:
   ```bash
   git checkout main
   # Edit CMakeLists.txt: MINOR = Y+1, PATCH = 0
   git add CMakeLists.txt
   git commit -m "Begin X.(Y+1).0 development cycle"
   ```

5. Push everything:
   ```bash
   git push origin release/X.Y.x
   git push origin main
   git push origin vX.Y.0
   ```

### SparseDiffPy

6. **Wait for SparseDiffEngine CI to pass.** Check the workflow run at
   the [Actions tab](https://github.com/SparseDifferentiation/SparseDiffEngine/actions).
   Do not proceed until the Engine's GitHub Release has been created
   successfully.

7. From `main`, create a release branch and point the submodule at
   the Engine's tagged release:
   ```bash
   git checkout -b release/X.Y.x
   cd SparseDiffEngine
   git checkout vX.Y.0
   cd ..
   ```

8. Set the version to `X.Y.0` in `pyproject.toml`, commit, and tag:
   ```bash
   # Edit pyproject.toml: version = "X.Y.0"
   git add SparseDiffEngine pyproject.toml
   git commit -m "Release vX.Y.0"
   git tag vX.Y.0
   ```

9. Lay groundwork for the next patch. Bump version to `X.Y.1` and commit.
   *Do not* tag this commit:
   ```bash
   # Edit pyproject.toml: version = "X.Y.1"
   git add pyproject.toml
   git commit -m "Begin X.Y.1 development"
   ```

10. Switch to `main` and bump to the next minor version for development.
    Also update the submodule to track Engine's `main`:
    ```bash
    git checkout main
    cd SparseDiffEngine
    git checkout main
    cd ..
    # Edit pyproject.toml: version = "X.(Y+1).0"
    git add SparseDiffEngine pyproject.toml
    git commit -m "Begin X.(Y+1).0 development cycle"
    ```

11. Push everything:
    ```bash
    git push origin release/X.Y.x
    git push origin main
    git push origin vX.Y.0
    ```

## Patch Release (e.g., 0.2.1)

A patch release increments the PATCH version on an existing release branch.

### SparseDiffEngine

1. Checkout the release branch and cherry-pick fixes from `main`:
   ```bash
   git checkout release/X.Y.x
   git cherry-pick <commit-hash>
   ```

2. Set the version to `X.Y.Z` in `CMakeLists.txt`, commit, and tag:
   ```bash
   # Edit CMakeLists.txt: PATCH = Z
   git add CMakeLists.txt
   git commit -m "Set version to X.Y.Z for release"
   git tag vX.Y.Z
   ```

3. Lay groundwork for the next patch. Bump PATCH to Z+1, commit (no tag):
   ```bash
   # Edit CMakeLists.txt: PATCH = Z+1
   git add CMakeLists.txt
   git commit -m "Begin X.Y.(Z+1) development"
   ```

4. Push:
   ```bash
   git push origin release/X.Y.x
   git push origin vX.Y.Z
   ```

### SparseDiffPy

5. **Wait for SparseDiffEngine CI to pass.** Check the workflow run at
   the [Actions tab](https://github.com/SparseDifferentiation/SparseDiffEngine/actions).
   Do not proceed until the Engine's GitHub Release has been created
   successfully.

6. Checkout the release branch, cherry-pick or update bindings as needed,
   and point the submodule at the Engine's new tag:
   ```bash
   git checkout release/X.Y.x
   cd SparseDiffEngine
   git checkout vX.Y.Z
   cd ..
   ```

7. Set the version to `X.Y.Z` in `pyproject.toml`, commit, and tag:
   ```bash
   # Edit pyproject.toml: version = "X.Y.Z"
   git add SparseDiffEngine pyproject.toml
   git commit -m "Release vX.Y.Z"
   git tag vX.Y.Z
   ```

8. Lay groundwork for the next patch. Bump version to `X.Y.(Z+1)`, commit (no tag):
   ```bash
   # Edit pyproject.toml: version = "X.Y.(Z+1)"
   git add pyproject.toml
   git commit -m "Begin X.Y.(Z+1) development"
   ```

9. Push:
   ```bash
   git push origin release/X.Y.x
   git push origin vX.Y.Z
   ```

## Deployment

Deployments are automatically triggered by pushing a version tag (`v*`).

**SparseDiffEngine** (`.github/workflows/release.yml`):
- Runs tests on Ubuntu and macOS
- Creates a GitHub Release with auto-generated release notes

**SparseDiffPy** (`.github/workflows/build-and-publish.yml`):
- Builds wheels for Python 3.11-3.14 on Ubuntu, macOS, and Windows
- Publishes to TestPyPI first, then to PyPI

Progress can be monitored from the Actions tabs:
- [SparseDiffEngine Actions](https://github.com/SparseDifferentiation/SparseDiffEngine/actions)
- [SparseDiffPy Actions](https://github.com/SparseDifferentiation/SparseDiffPy/actions)

After a successful deployment, verify the package on
[PyPI](https://pypi.org/project/sparsediffpy/).

### Example: deploying v0.2.0

After pushing the `v0.2.0` tag in SparseDiffEngine:

1. Go to [SparseDiffEngine Actions](https://github.com/SparseDifferentiation/SparseDiffEngine/actions).
   A workflow run labelled `v0.2.0` should appear. It will:
   - Run `all_tests` on Ubuntu and macOS
   - If both pass, create a GitHub Release at
     https://github.com/SparseDifferentiation/SparseDiffEngine/releases/tag/v0.2.0
     with auto-generated release notes

2. Once the Engine release succeeds, push the `v0.2.0` tag in SparseDiffPy.
   Go to [SparseDiffPy Actions](https://github.com/SparseDifferentiation/SparseDiffPy/actions).
   A workflow run labelled `v0.2.0` should appear. It will:
   - Build wheels for every Python 3.11-3.14 + OS combination
   - Publish all wheels and the sdist to
     [TestPyPI](https://test.pypi.org/project/sparsediffpy/)
   - After TestPyPI succeeds, publish to
     [PyPI](https://pypi.org/project/sparsediffpy/)

3. Verify the release landed:
   ```bash
   pip install sparsediffpy==0.2.0
   python -c "import sparsediffpy; print('OK')"
   ```

If the workflow fails intermittently (e.g., network timeouts), re-run it
from the Actions tab. If code changes are required, you will need to
delete the tag, fix the issue, re-tag, and push again.

## Notes

- Always release SparseDiffEngine first — SparseDiffPy depends on it
  via the git submodule.
- The submodule pointer in SparseDiffPy must point at the exact tagged
  commit in SparseDiffEngine.
- Push branches before tags so CI has access to the branch context.
- Never tag a commit that has not been pushed to the remote.
