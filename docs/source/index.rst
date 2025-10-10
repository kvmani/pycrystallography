##.. image:: _static/pycrystallography.png
##   :width: 300 px
##   :alt: pycrystallography
##   :align: center

Getting started
---------------

The new command line interface is exposed via the ``pcg`` console script. Validate the
example Burgers orientation relation configuration::

   pcg config validate --config ../examples/burgers_zr.yaml --print

Generate a composite SAED pattern (dry run)::

   pcg composite tem --relation burgers-zr --config ../examples/burgers_zr.yaml --dry-run

API documentation
-----------------

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

