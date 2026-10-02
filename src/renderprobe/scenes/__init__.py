"""Scene generators.

Every scene here is part of the product: it has been through `renderprobe validate` and
through the calibration gate in `tools/calibrate.py`, so a model's accuracy on it sits
somewhere the three-way decomposition can say something about.

The scenes that back the offline test suite live in `tests/fixtures/` instead. They were
written before the calibration gate and were never taken through it, so a number measured
on one means nothing about a model; keeping them out of the installed package is what
stops them reading as part of the instrument.
"""
