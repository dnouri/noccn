`noccn` is a collection of wrappers around Alex Krizhevsky's
`cuda-convnet <http://code.google.com/p/cuda-convnet/>`_.

What is cuda-convnet?
=====================

  It is a fast C++/CUDA implementation of convolutional (or more
  generally, feed-forward) neural networks. It can model arbitrary
  layer connectivity and network depth. Any directed acyclic graph of
  layers will do. Training is done using the back-propagation
  algorithm.

What is noccn for then?
=======================

`noccn` adds a few scripts and wraps existing `cuda-convnet` scripts
to allow for defining parameters on the filesystem.  Scripts in
`noccn` will always refer to an `options.cfg` file that includes all
the parameters usually passed to `cuda-convnet` scripts on the
command-line.

`noccn` is being developed but hasn't reached a stable version yet.

The options.cfg file
--------------------

Here how an `options.cfg` file can look like::

  #!ini
  [DEFAULT]
  data-provider = convdata.CIFARDataProvider
  include = $HERE/../options.cfg

  [train]
  layer-def = $HERE/layers.cfg
  layer-params = $HERE/layer-params.cfg
  data-path = $HERE/batches/
  save-path = $HERE/tmp/
  train-range = 20-50
  test-range = 1-9
  convnet.give_up_epochs = 100

  [show]
  test-range = 1

  [predict-test]
  train-range = 1
  test-range = 1-9
  report = 1

  [predict-valid]
  train-range = 1
  test-range = 10-19
  report = 1

  [predict-train]
  train-range = 1
  test-range = 20-50
  report = 1

  [predict-100-125]
  train-range = 1
  test-range = 100-125
  write-preds-cols = 1
  multiview-test = 1
  logreg-name = logprob
  write-preds = $HERE/preds-100-125.csv

The section ``[train]`` contains all the parameters for training
(`ccn-train`).  Similarly, ``[show]`` has all the parameters for the
`ccn-show` script and so on.  We can define multiple sections for the
`ccn-predict` script.

The section ``[DEFAULT]`` defines variables that are used for all
other sections.  The ``data-provider`` is a dotted path to the data
provider implementation that you want to use.  It may include a
``include`` parameter to include shared parameters from another file.

Scripts
-------

Scripts included in `noccn` resemble those found in `cuda-convnet`
itself.  With the exception of `ccn-predict`, of which there's no
equivalent in `cuda-convnet`.  `ccn-predict` is used to, given a
pickled model and input batches, predict outputs to either generate a
report (``report = 1``), or generate a CSV file with the predictions
made (``write-preds = filename``).  When running the `ccn-predict`
script, make sure you pass the model snapshot using the ``-f``
argument on the command-line.

``ccn-train`` extends the train script in `cuda-convnet` to allow for
saving a model snapshot only when it has achieved a lower error on the
test set.  (The original script would save the state regardless of
whether performance improved or not.)  You can also use
``convnet.give_up_epochs = 100`` parameter to control after how many
epochs without an improvement on the test error the train script
should quit training.  These two features are particularly useful if
you want to run a few model configurations unattended.
