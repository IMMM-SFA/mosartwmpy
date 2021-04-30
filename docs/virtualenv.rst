Python Virtual Environments
===========================

Maintaining different versions of Python can be a chore, especially when ``mosartwmpy`` is forcing you to use Python 3.9+ 😈. There are many tools for managing Python environments, and many more will probably be developed after I write this, but at this moment in time my two recommendations are:

* `PyCharm <https://www.jetbrains.com/pycharm/>`_ IDE -- this is great for developing code in Python, and can automatically create virtual environments for a codebase by detecting versions and dependencies from the ``setup.py`` or ``setup.cfg``.
* `pyenv <https://github.com/pyenv/pyenv>`_ CLI -- this is a shell based tool for installing and switching between different versions of Python and dependencies. I will give a brief tutorial of using ``pyenv`` below, but recognize that the instructions may change over time so the ``pyenv`` documentation is the best place to look.

To create a Python 3.9 virtual environment, try the following steps:

* Install pyenv:

  - if on Mac, use `brew <https://brew.sh/>`_: brew install pyenv
  - if on a linux system, try `pyenv-installer <https://github.com/pyenv/pyenv-installer>`_
  - if on Windows, try `pyenv-win <https://github.com/pyenv-win/pyenv-win>`_

* Install Python 3.9:

  - in a shell, run ``pyenv install 3.9.1``

* Activate Python 3.9 in the current shell

  - in the shell, run ``pyenv shell 3.9.1``

* Proceed with the install of mosartwmpy:

  - in the same shell, run ``pip install mosartwmpy``

* Now you can interact with ``mosartwmpy`` in this current shell session

  - if you start a new shell session you will need to run ``pyenv shell 3.9.1`` again before proceeding
  - this new shell session should maintain all previously pip installed modules for Python 3.9.1