# How to make a new public release for ELEKTRONN


### Requirements:

1. Python 2.7.x with virtualenv >= 15.0 (older versions may work, but they are untested).
2. [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html),
   with its `bin/` directory prepended to your shell's $PATH. If conda is not in your $PATH,
   you can set it up temporarily by running

        $ PATH="<anaconda-root>/bin:$PATH"
          # e.g. $ PATH="~/anaconda2/bin:$PATH"

3. The Anaconda environment needs the conda-build package. Install it via

        $ conda install conda-build

4. For conda envs: a running bash or zsh session (for other shells, see https://github.com/conda/conda/blob/master/shell/README.md)
All commands prefixed with "$" signs should be entered in bash or zsh.
5. `~/.pypirc` configured according to https://docs.python.org/2/distutils/packageindex.html#the-pypirc-file


### Test:

        $ cd <elektronn-source-root>
        $ python2 -m virtualenv /tmp/venv2a
        $ source /tmp/venv2a/bin/activate
        $ python2 setup.py check --restructuredtext --strict
          # (PyPi won't render RSTs with any warnings. README.rst has to be perfect.)
        $ python2 -m pip install .
        $ elektronn-train MNIST_CNN_warp_config.py # make sure training works
        $ deactivate
        
        $ conda build .
        $ conda create -p /tmp/cenv2a
        $ source activate /tmp/cenv2a
        $ conda install --use-local elektronn
        $ elektronn-train MNIST_CNN_warp_config.py
        $ source deactivate
        
        # Remove the new virtualenvs and conda envs after you are done testing:
        $ rm -rf /tmp/venv2a
        $ rm -rf /tmp/cenv2a


### Release:
1. Update version strings in "setup.py" and in "meta.yaml" to "`<major>.<minor>.<fix>`", e.g. "1.1.2" and commit
   changes ([example](https://github.com/ELEKTRONN/ELEKTRONN/commit/1d5d0cbd805eeb843471b5309e4b623c201d7969)).
2. Go to https://github.com/ELEKTRONN/ELEKTRONN/releases and click "Draft a new release".
3. Enter "Tag version" in the format "`v<major>.<minor>.<fix>`", e.g. "v1.1.2". (Mind the "v" prefix!)
4. Publish release.
5. Freshly `git clone` the repo to a new working directory (do not just `git pull`, that could mess things up):

        $ git clone https://github.com/ELEKTRONN/ELEKTRONN.git elektronn-release
        $ cd elektronn-release
        $ git checkout <version-tag>
          # where <version-tag> is the string you entered in step 3.

6. Upload to pypi (**Attention: This can not be undone.
   You will never be able to upload a package with the same version string again!**):

        $ python2 setup.py sdist upload -r pypi
          # result should be "Server response (200): OK".

7. Update the "elektronn" conda-forge feedstock:

    * Fork https://github.com/conda-forge/elektronn-feedstock
    * Create a PR, changing necessary fields in recipe/meta.yaml
    * (If necessary, install/update `conda-smithy`, run `$ conda smithy rerender`, commit and push.)
    * If CI is sucessful, merge PR.

8. Verify the published packages:

        $ python2 -m virtualenv /tmp/venv2b
        $ source /tmp/venv2b/bin/activate
        $ python2 -m pip install elektronn
        $ elektronn-train MNIST_CNN_warp_config.py
        $ deactivate
        
        $ conda create -p /tmp/cenv2b
        $ source activate /tmp/cenv2b
        $ conda install -c conda-forge elektronn
        $ elektronn-train MNIST_CNN_warp_config.py
        $ source deactivate
        
        # Remove the new virtualenvs and conda envs after you are done testing:
        $ rm -rf /tmp/venv2b
        $ rm -rf /tmp/cenv2b

9. Update AUR package (https://aur.archlinux.org/packages/elektronn/):
    * See https://wiki.archlinux.org/index.php/Arch_User_Repository#Creating_a_new_package for initial setup
    * and https://wiki.archlinux.org/index.php/Arch_User_Repository#Updating_packages for regular updates.
    * If there are dependency or packaging changes, you should also update https://aur.archlinux.org/packages/elektronn-git/
