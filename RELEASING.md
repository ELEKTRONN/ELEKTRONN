# How to make a new public release for ELEKTRONN


### Requirements:

1. Existing installation of python2.7 with setuptools 18.0 or higher
2. Anaconda or Miniconda installed under the `<anaconda-root>` directory.
3. The Anaconda environment needs the conda-build and anaconda-client packages. Install them via

        $ conda install conda-build anaconda-client


### Test:
1. Make new virtualenvs and conda envs, `cd` to them and run:

        $ cd <elektronn-source-root>
        $ source <virtualenv2-root>/bin/activate
        $ python2 -m pip install .
        $ elektronn-train MNIST_CNN_warp_config.py # make sure training works

        $ source <anaconda-root>/bin/activate <new-env>
          # e.g. source activate ~/anaconda2/bin/activate root
        $ conda install .
        $ elektronn-train MNIST_CNN_warp_config.py

2. Erase these new environments. They are only confusing if they stay around.


### Release:
1. Update version strings in "setup.py"" and in "meta.yaml" to "`<major>.<minor>.<fix>`", e.g. "1.1.2" and commit changes ([example](https://github.com/ELEKTRONN/ELEKTRONN/commit/1d5d0cbd805eeb843471b5309e4b623c201d7969)).
2. Go to https://github.com/ELEKTRONN/ELEKTRONN/releases and click "Draft a new release".
3. Enter "Tag version" in the format "`v<major>.<minor>.<fix>`", e.g. "v1.1.2". (Mind the "v" prefix!)
4. Publish release.
5. Freshly `git clone` the repo to a new working directory (do not just `git pull`, that could mess things up):

        $ git clone https://github.com/ELEKTRONN/ELEKTRONN.git elektronn-release
        $ cd elektronn-release

6. Upload to pypi (**Attention: This can not be undone. You will never be able to upload a package with the same version string again!**):

        $ python2 setup.py sdist upload -r pypi
          # result should be "Server response (200): OK".

7. Upload to anaconda.org:
In a bash-compatible shell (not tcsh or fish or xonsh etc.) run

        $ source <anaconda-root>/bin/activate <env>
        $ anaconda login
          # (only necessary if you are not already logged in)
          # enter your private username, e.g xeray or mdraw. Do not enter elektronn.
          # (your username has to be registered under the elektronn organization though.)
        $ conda build .
        $ anaconda upload --user elektronn <path-to-elektronn.tar.bz2>
          # You find the <path-to-elektronn.tar.bz2> argument by looking at the last lines of "conda build ."'s output.
          # (Do not just copy-paste the proposed command. The "--user elektronn" flag is needed.)
8. Test everything again as if you were a new user: Install ELEKTRONN in a clean virtual environment and run some tests:

        # Make and change to new virtualenvs and conda envs and run:
        $ source <virtualenv2-root>/bin/activate
        $ python2 -m pip install elektronn
        $ elektronn-train MNIST_CNN_warp_config.py

        $ source <anaconda-root>/bin/activate <new-env>
        $ conda install elektronn
        $ elektronn-train MNIST_CNN_warp_config.py

        # Remember to remove the new virtualenvs and conda envs after you are done testing.

9. Update AUR package (https://aur.archlinux.org/packages/elektronn/):
    * See https://wiki.archlinux.org/index.php/Arch_User_Repository#Creating_a_new_package for initial setup
    * and https://wiki.archlinux.org/index.php/Arch_User_Repository#Updating_packages for regular updates.
    * If there are dependency or packaging changes, you should also update https://aur.archlinux.org/packages/elektronn-git/
