# Installation

```{note}
PassengerSim is currently available only for Mac and Linux operating systems.
Support for other operating systems is not planned for the near future.
```

## Before you begin: install `uv` if needed

PassengerSim is a Python package that is most easily installed in an environment
managed by `uv`.  To check if you have `uv` installed, open a terminal
and type `uv --version`.  If you see a version number, you have `uv`.

If you have `uv`, but with an outdated version (i.e. less than 0.11), you might be
able to update it by running `uv self update`. If that doesn't work (it may not,
depending on how you originally installed `uv`), you'll probably see a hint for
the right command to do so.

If you don't have `uv`, you can install it by following the instructions on
the [UV Docs](https://docs.astral.sh/uv/getting-started/installation/).


## Installing PassengerSim

If you have received a file named something like `install-passengersim-mac-v0.0-20240101.sh`,
you can use it to install PassengerSim in a `uv` virtual environment.  To do so, open a terminal
and navigate to the directory where the file is located.  Then, run the following command:

```shell
bash install-passengersim-mac-v0.0-20240101.sh
```

!!! tip

    Don't copy-paste the command above, as the exact filename of the installation script
    will vary based on the version of PassengerSim and the release data.  Instead, use the
    filename of the file you have received.

If you use the installation script, it will create a new `uv` virtual environment
named `.paxlab` and install both PassengerSim and Jupyter Lab into that environment.  After
the installation is complete, you can start a Jupyter Lab session from the environment by
running `./start-paxlab`.

Alternatively, you may have received set of wheel files, with the extension ".whl".  You
can install these into your Python environment using `pip` or `uv pip`.  Doing so is just
like installing any other Python package using pip, except that you give the filename of
the wheel file instead of a package name.  For example, if you have a wheel file named
`passengersim-0.0-py3-none-any.whl`, you can install it by running the following command:

```shell
uv pip install ./passengersim-0.0-py3-none-any.whl
```

The "./" at the beginning of the filename is important, as it tells `pip` or `uv` to look
for the file in the current directory, not on the internet. If you install from wheels,
you are responsible for keeping track of which environment you installed into (virtual or
otherwise), so you can be sure to activate that environment when you want to use PassengerSim.



## Alternative Installation: Docker

Do you have a link to a `sky` docker image file? If not, these are not the install
instructions you are looking for. If you do have a link, go ahead and download
the file.  If you are installing on Windows or an Intel-based Mac, you want the
amd64 image.  For installing on more recent Apple Silicon-based Macs, use an
arm64 image.

In addition to the image file, you'll want to download and install
[Docker Desktop](https://www.docker.com) for your computer.  Once it's installed, make
sure it's running (you should be able to open and see Docker's dashboard).

Then, open a console terminal and change to the directory where the docker image
`sky.tar` file has been saved.  It might be named something slightly different from
`sky.tar`, in which case change the filename in the commands you type to match
the filename of the image file you have.  The first step is to run

```shell
docker load --input sky.tar
```

This command may take a few moments to complete, as the content of the downloaded
image file will be imported into docker.  You'll need several free gigabytes of
hard disk space to complete this step, so if you encounter unexplained EOF errors
here that may be the problem.

After the docker image is loaded, you should see a `sky` image in the docker desktop
dashboard.  If you can see it, you should then be able to start a container with
this image using the following command:

=== "Mac/Linux"

    ``` bash
    docker run -p 8899:8899 --rm --volume "$(pwd)":/tmp/workplace/work sky:latest
    ```

=== "Windows"

    ``` bash
    docker run -p 8899:8899 --rm --volume "%cd%":/tmp/workplace/work sky:latest
    ```

Within this command, we have:

- `-p 8899:8899` tells docker to expose the container's port 8899 (which has been
  configured to be the port served by Jupyter Lab) to localhost.
- `--rm` means to remove the container when it exits, so there isn't an
  extraneous container image file left on your file system.
- `--volume "...":/tmp/workplace/work` makes the current working directory
  available inside Jupyter Lab in a directory named `work` (the container has
  been configured with `/tmp/workplace` as the base location for Jupyter).
- `sky:latest` tells Docker to use the latest version of the user image
  that was installed in the `docker load` step above.  If your installed
  docker image (as shown in the Docker Dashboard) has a different name than
  `sky` then change the command to give the corrent image name.
