.. |badge_tests| image:: https://github.com/sourceperl/pyModbusTCP/actions/workflows/tests.yml/badge.svg?branch=master
                :target: https://github.com/sourceperl/pyModbusTCP/actions/workflows/tests.yml

.. |badge_docs| image:: https://readthedocs.org/projects/pymodbustcp/badge/?version=latest
               :target: http://pymodbustcp.readthedocs.io/en/latest/?badge=latest

pyModbusTCP |badge_tests| |badge_docs|
======================================

A simple Modbus/TCP client library for Python.

Since version 0.1.0, a server is also available for test purpose only (don't use in project).

pyModbusTCP is pure Python code without any extension or external module
dependency.

Tests
-----

The module is currently test on Python 3.5, 3.6, 3.7, 3.8, 3.9 and 3.10.


Setup
-----

You can install this package from:

PyPI, the easy way:

.. code-block:: bash

    sudo pip install pyModbusTCP

GitHub:

.. code-block:: bash

    git clone https://github.com/sourceperl/pyModbusTCP.git
    cd pyModbusTCP
    sudo python setup.py install

GitHub with pip:

.. code-block:: bash

    sudo pip install git+https://github.com/sourceperl/pyModbusTCP.git

Usage example
-------------

See examples/ for full scripts.

include (for all samples)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyModbusTCP.client import ModbusClient

module init (TCP always open)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # TCP auto connect on first modbus request
    c = ModbusClient(host="localhost", port=502, unit_id=1, auto_open=True)

module init (TCP open/close for each request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # TCP auto connect on modbus request, close after it
    c = ModbusClient(host="127.0.0.1", auto_open=True, auto_close=True)

Read 2x 16 bits registers at modbus address 0 :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    regs = c.read_holding_registers(0, 2)
    if regs:
        print(regs)
    else:
        print("read error")

Write value 44 and 55 to registers at modbus address 10 :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    if c.write_multiple_registers(10, [44,55]):
        print("write ok")
    else:
        print("write error")

Documentation
-------------

Documentation available online at http://pymodbustcp.readthedocs.io/.

Know issue for older releases (v0.1.x) that support python 2
------------------------------------------------------------

On windows OS with older Python version (<3), win_inet_pton module is require. This avoid exception "AttributeError:
'module' object has no attribute 'inet_pton'".

install win_inet_pton:

.. code-block:: bash

    sudo pip install win_inet_pton

import win_inet_pton at beginning of your code:

.. code-block:: python

    import win_inet_pton
    from pyModbusTCP.client import ModbusClient
