Overview
========

Recurrent Neural Networks (RNNs) are seen in a variety of machine learning/AI applications.
This discussion will attempt to focus on the core ideas, yet provide some practical implementation
examples and guidance.

The discussion will begin with some basic ideas about what recurrence is and connect this
to a more general framework from a signal processing basis.  Then, some common implementations
will be introduced with examples, to gain some familiarity with RNNs in practice.

The next section will provide some design guidance as well as some analysis practices.


Basic recurrance architecture
-----------------------------

To begin with, let us consider a basic recurrant architecture (RNN).  The most simple version is a single node with an input that is derived from its output as follows:

.. image:: graphics/rec_network.svg

In the above figure, we see the main components which permeate the discussion about RNNs.  Namely, we have an input channel which is summed with a channel that is derived from the output of the network.  Typically, each of these channels will have an adjustable weight that is learned during the training process.  Because we are considering this network in the context of a digital computer, we will have values at discrete time points.  It is useful to imagine that the data value of the recurrent channel is stored in a memory location whose contents is copied just before a new function evaluation is made at a given time step.  Thus, when the value changes as a consequence, only the memory location will be updated, not the copy.  Therefore the network will attain a stable output immediately.  Another way to say this is that we are effectively introducing a unit time delay in the recurrant (feedback) channel and will keep this in mind implicitly in all our further discussions.
