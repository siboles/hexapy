Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Anaconda
--------

hexapy is designed for use with `Anaconda <https://www.continuum.io/downloads>`_ Python from Continuum Analytics. You can install either the full Anaconda package or Miniconda.

To install the current release
------------------------------

.. code-block:: guess

   conda install -c siboles hexapy

To install the latest version
-----------------------------

The source code can be downloaded from `GitHub <https://github.com/siboles/hexapy/archive/master.zip>`_ or if git is installed cloned with:

.. code-block:: guess

   git clone https://github.com/siboles/hexapy.git

The module can then be installed following the Standard Python instructions below.

Standard Python
---------------

hexapy can be installed for a standard Python environment from source.

Navigate to src/ in the source code directory tree and type:

.. code-block:: guess

   python setup.py install

This may require sudo priviliges in Linux environments depending on where Python is installed. Alternatively, this can be done in a virtual environment.

numpy and scipy must be installed.
