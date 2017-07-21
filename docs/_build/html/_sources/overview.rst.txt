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

.. image:: 14by14axes.jpg

