# XPyriments: Automating your experimentation with PyCaret

---
**DISCLAIMER**: This notebook does not intend to achieve a model with great results.

**DISCLAIMER 2**: This was prepared using PyCaret 2.3.6, before that the OOP version was released.

---

## Why do I need this?

PyCaret does an amazing job logging all the metrics from an experiment into mlflow. If you use a different experiment tracking tool, however, you need to improvise.
This is my way of allowing PyCaret to log experiments into a tracker which stores metrics from standard output.

## Structure

Inside the `xpyriments` module there are two directories: `utils` and `config`.
In `utils`, there are a few scripts divided by topic that you can use when running an experiment. Whereas in `config`, there are fixed values one can change to better describe your experiment.

## Usage

The main usage of XPyriments is through the class `Experiment` located in `xpyriments.utils.experiment`.

## Contribution
You can clone/fork this repo as you wish.

If this was useful to you, please give a star! And also, open issues, and collaborate away!
