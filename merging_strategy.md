# LlamaCPP Fork Management Strategy

## Problem Statement

The team maintains a fork of llama.cpp with custom modifications but needs to stay synchronized with the rapidly evolving upstream repository. Traditional PR-based merging creates merge commits or squashes that modify git history, making future upstream synchronization difficult.

## Proposed Solution

### Branch Structure

- **temp-upstream**: Tracks the official upstream llama.cpp repository (pure upstream)
- **temp-latest**: Contains team's custom changes rebased on top of upstream
- **Feature branches**: Built on top of temp-latest for new changes

### Synchronization Process

#### Initial Setup (one-time)
To make things clear, will start from a blank state. You can skip steps you've already done:

1. **Fork the repository**: Fork `qvac-ext-lib-llama.cpp` repo in GitHub (e.g., `https://github.com/olyasir/qvac-ext-lib-llama.cpp`)

2. **Clone locally**:
   ```bash
   git clone git@github.com:olyasir/qvac-ext-lib-llama.cpp.git
   cd qvac-ext-lib-llama.cpp
   ```

3. **Configure remotes**:
   ```bash
   # Add upstream ggml remote
   git remote add ggml git@github.com:ggml-org/llama.cpp.git
   git fetch ggml
   
   # Add tether remote
   git remote add tether git@github.com:tetherto/qvac-ext-lib-llama.cpp.git
   git fetch tether
   ```

#### Regular Synchronization Process

1. **Prepare temp-latest branch**:
   ```bash
   git checkout temp-latest
   git pull
   ```

2. **Tag the branch if no existing tag yet**:
   ```bash
   git tag v6469.1.2
   git push tether tag v6469.1.2
   ```

REVIEW AND DECIDE: The following step combines all commits since the last rebase to minimize manual resolution of conflicts. This makes sure that any related changes and/or reverts are only applied once, resulting in a more direct and simple rebase process. The disadvantage of this process is that individual commits are lost and replaced by one big commit message instead.

3. **Squash all commits since the last rebase**:
   ```bash
   # get all commits since current branch diverged with the upstream updates (b7028 in this case)
   git log $(git merge-base HEAD b7028)..HEAD > commit_messages.txt

   # combine all commits into a single one and aggregate all commit messages
   git reset --soft $(git merge-base HEAD b7028)
   git commit -F commit_messages.txt
   ```

4. **Rebase onto new upstream tag** (e.g., b7028):
   ```bash
   git rebase b7028
   ```

5. **Resolve conflicts** Git will stop if it finds conflicts. Resolve them as appropriate (may need to check with original writers). To assist with the conflict resolution process, it is better and recommended to have the common ancestor’s code viewable for better context. Set with the following git config:
   ```bash
   git config --global merge.conflictstyle diff3
   ```

A conflict would now look like this:
   ```bash
   <<<<<<< current
   /* the current ggml upstream changes  */
   ||||||| base
   /* the original code from the common ancestor */
   =======
   /* incoming tether custom changes */
   >>>>>>> incoming
   ```

6. **Push rebased changes to forked repo**:
   ```bash
   # safer than git push -f
   git push origin --force-with-lease
   ```

7. **Create and push new tag**:
   ```bash
   # add description like "Sync with upstream version b7028"
   git tag -a v7028.0.0 -m "Sync with upstream version b7028"
   git push origin tag v7028.0.0
   ```

8. **Test and publish**: Test the new tag.

#### Testing Process

1. **Get and extract test project**: Download the test project from [vcpkg-test-llama-cpp](https://drive.google.com/file/d/1Fm47_QsPsjp-kjPnQpQiRTE5KIrxMh_G/view?usp=sharing) (simple project that depends on the llama-cpp port)

2. **Fork the qvac registry**: Fork `qvac-registry-vcpkg` repo in GitHub

3. **Clone locally**:
   ```bash
   git clone git@github.com:olyasir/qvac-registry-vcpkg.git
   ```

4. **Update vcpkg port**:
   ```bash
   # copy latest ports/llama-cpp folder from qvac-registry-vcpkg
   cd vcpkg-test-llama-cpp
   cp -r ../qvac-registry-vcpkg/ports/llama-cpp vcpkg/ports/llama-cpp
   ```

5. **Update version**: In `vcpkg/ports/llama-cpp/vcpkg.json`, update version number to new tag (without 'b' prefix)
   - For tag `b6789.0.0` → version should be `6789.0.0`

6. **Install dependencies as needed**:
   ```bash
   # for debian-based linux
   sudo apt install -y libvulkan-dev glslc
   ```

7. **Initial build attempt**:
   ```bash
   bare-make generate
   ```

8. **Fix SHA512 hash**: Configuration will fail with hash mismatch error:
   ```
   error: download from https://github.com/tetherto/qvac-ext-lib-llama.cpp/archive/b6435.0.0.tar.gz had an unexpected hash
   note: Expected: 9baedc3c4ff681222d8fe16ac10346af9cd7fd5a4a6047448f8a3ad0712ba8e35dbd06af16b3a8c6c8b25518b43fd3b152275e90969f0c33cf851cdb44484eb0
   note: Actual  : c869a45e809c367cae6122bfc26c26f16767b010f2da804eb6d20eab8fc9ee8a6fa9c35d04792d0dc1e7483a1b552441027a96ebd30cfb8ac455a3da52801f59
   ```
   Update `vcpkg/ports/llama-cpp/portfile.cmake` - replace SHA512 line with the "Actual" value

9. **Final verification**:
   ```bash
   bare-make generate
   bare-make build
   ```
   If successful, the sync worked properly and you can now request someone from the team to review your work.

#### Pushing updates

1. **Push rebased changes to tether repo**:
   ```bash
   # safer than git push -f
   git push tether --force-with-lease
   ```

2. **Update vcpkg registry with the modified llama-cpp ports**:
   ```bash
   # copy over the changes
   cd ../qvac-registry-vcpkg
   cp -r ../vcpkg-test-llama-cpp/vcpkg/ports/llama-cpp ports/

   # commit the changes
   git add -u
   git commit -m "Updated llama-cpp to v7028.0.1"
   ```

3. **Update the versions database**:
   ```bash
   vcpkg --x-builtin-ports-root=./ports --x-builtin-registry-versions-dir=./versions x-add-version --verbose llama-cpp
   git add -u
   git commit -m "Updated llama-cpp to v7028.0.1" --amend # REVIEW: can we do an amend here? Or do we really need to do two distinct commits?
   git push origin
   ```

4. **Create a github pull request**


### Version Management
*(on temp-latest branch)*

- Base versions follow upstream tags (e.g., b5932)
- Extended versions add incremental numbers:
  - **b5932.0.0**: temp-upstream + mtmd changes
  - **b5932.1.0**: temp-upstream + mtmd + load-from-buffer changes
  - And so on...

## Development Workflow

1. **New PRs**: Create against temp-latest (which contains existing custom changes)
2. **After synchronization**: New PRs pointed to temp-latest should include all our changes and the new upstream version
3. **vcpkg integration**: vcpkg registry points to specific commit hashes, not branch names
4. **Testing**: Test new tags before publishing to vcpkg

## Benefits

This strategy ensures the team can maintain their custom modifications while staying current with upstream llama.cpp development.

- Maintains clean git history aligned with upstream
- Enables easy future synchronization with upstream
- Protects against accidental direct merges to main
- Allows multiple teams (including external collaborators like Collabora) to build on top of stable versions
- Custom changes accumulate incrementally while staying current with upstream
