noccn is a collection of wrappers around Alex Krizhevsky's
`cuda-convnet <http://code.google.com/p/cuda-convnet/>`_.

What is cuda-convnet?
=====================

According to its website, `cuda-convnet` is "a fast C++/CUDA
implementation of convolutional (or more generally, feed-forward)
neural networks. It can model arbitrary layer connectivity and network
depth. Any directed acyclic graph of layers will do. Training is done
using the back-propagation algorithm."

`cuda-convnet` has really `nice docs on its homepage
<http://code.google.com/p/cuda-convnet/>`_.

What is noccn for then?
=======================

noccn helps you deal with cuda-convnet's many command-line parameters
by allowing you to put them into a configuration file (usually
`options.cfg`).  noccn also allows you to specify in your
configuration file how you're building your data batches.  This way,
you'll easily remember how exactly you ran your experiments, and how
you got your results.

There is support for turning a list of folders containing images into
batches.  The batch creation code can be extended with your own batch
creator.

noccn is fairly stable -- I use it quite a lot -- but it's still
underdocumented.  A lot of the options will however just map to
cuda-convnet's own.

The options.cfg file
--------------------

Here's an example of an `options.cfg` file::

  #!ini
  [DEFAULT]
  data-provider = convdata.CIFARDataProvider
  include = $HERE/../defaults.cfg

  [train]
  layer-def = $HERE/layers.cfg
  layer-params = $HERE/layer-params.cfg
  data-path = $HERE/../../batches/
  train-range = 1-29
  test-range = 30-40
  save-path = $HERE/tmp
  give-up-epochs = 200

  [show]
  test-range = 41-44

  [predict-test]
  train-range = 1
  test-range = 30-40
  report = 1

  [predict-valid]
  train-range = 1
  test-range = 41-44
  report = 1

  [predict-train]
  train-range = 1
  test-range = 1-8
  report = 1
  # write-preds = $HERE/preds/preds-train.csv
  # write-preds-cols = 1

  [dataset]
  input-path = $HERE/../../images/
  pattern = *.jpg
  output-path = $HERE/../../batches/

The path to this `options.cfg` file is the first argument to every
script in noccn.  `options.cfg` and arguments on the command-line can
be combined, where arguments on the command-line will overrule those
in the config file.

The section ``[train]`` contains all the parameters for training
(`ccn-train`).  Similarly, ``[show]`` has all the parameters for the
`ccn-show` script and so on.  We can define multiple sections for the
`ccn-predict` script.

The section ``[DEFAULT]`` defines variables that are used for all
other sections.  The ``data-provider`` is a dotted path to the data
provider implementation that you want to use.  The default section may
have an ``include`` parameter to include shared parameters from
another file.

Installation in a virtualenv
----------------------------

Use `pip` to install `noccn` in a virtualenv::

  #!shell
  virtualenv noccn --system-site-packages
  cd noccn
  bin/pip install path/to/noccn

If you're on Debian or Ubuntu, you can install numpy and scipy like
this::



Scripts
-------

A few of the scripts included in noccn wrap those found in
`cuda-convnet` itself.  These are `ccn-train` and `ccn-show`.  Scripts
that noccn itself adds are `ccn-predict` and `ccn-make-batches`.

Some scripts require that you point them to a model snapshot or a
snapshot directory, using the `-f` argument.

ccn-train
~~~~~~~~~

Using `ccn-train` is simple; just pass the path to the `options.cfg`
file as defined above::

  #!shell
  bin/ccn-train models/01/options.cfg

noccn's train script will only save a snapshot if there was an
improvement in the test score.  If you want to store snapshots
regardless of whether or not the test score improved, you can pass
`always-save = 1`.

The `convnet.give_up_epochs` argument defines after how many epochs
without an improvement on the test score should we automatically stop
the learning.  This is useful if you want to run a few parameters
unattended.

ccn-show
~~~~~~~~

During training, you can take a look at the network's performance, at
random test samples and their predictions, and at the activations of
the first layer in your network using the `ccn-show` script::

  #!shell
  bin/ccn-show models/01/options.cfg -f models/01/tmp/ConvNet__*/

If you want to view a different convolutional layer, pass
e.g. `--show-filters=conv2`.

ccn-predict
~~~~~~~~~~~

The `ccn-predict` script prints out a classification report and a
confusion matrix.  This gives you numbers to evaluate your network's
performance::

  #!shell
  bin/ccn-predict models/01/options.cfg -f models/01/tmp/ConvNet__*/

ccn-make-batches
~~~~~~~~~~~~~~~~

The `ccn-make-batches` script is a handy way to create input batches
for use with `cuda-convnet` from a folder with images.  Within the
folder that you point `ccn-make-batches` to (through the
configuration's `[dataset]` section), you should have one folder per
category, with JPEG images belonging to that category inside.  The way
`ccn-make-batches` collects images can be configured through the
`collector` argument (default:
`noccn.dataset._collect_filenames_and_labels`).  The way input files
are converted to data vectors can be overridden by passing in a
different `creator` (default: `noccn.dataset.BatchCreator`).

An example::

  #!shell
  bin/ccn-make-batches models/01/options.cfg
