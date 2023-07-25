Results
=======

*Abstract*: *Not available yet*

Thesis
------

Download Thesis: *Not available yet*


Video Visualisations
--------------------

In the following, video visualisations of the results presented in the thesis are shown.
The input into the model is a line that is rotating counter-clockwise around the origin.
This input is fed into the sensory system that is shown in the center of the video.
The sensory system has 4 filters to extract different features. These features are visualised with different colors.

The output of the sensory system is fed into the S1, the network with lateral connections building net fragments.
These net fragments are visualised in the top right corner of the video, using the same colors per feature channel as the sensory system.

The net fragments are fed into S2, that maps the net fragments to a 1D representation of length 16. This 1D vector is visualised
as circles in the bottom center of the video. Thereby, a green circle corresponds to an active neuron, while a red circle corresponds
to an inactive neuron.
The reconstructed net fragment of S2 is shown in the bottom right corner of the video.

In the following, the results from multiple experiments are visualised. For each experiment, two videos are shown:
Both videos are produced by the same model using the same weights. However, the first
video shows the activations if a fixed threshold of 0.5 (S1) and 0.9 (S2) for each neuron is used (i.e. probabilities >=0.5 / >=0.9 lead to an activation of a neuron).
This helps to better understand the behaviour of the model.
The second video shows the activations if the neurons are sampled from a Bernoulli distribution.
This corresponds to the behaviour of the model during training.

Please not that the model has only seen horizontal, vertical, and diagonal lines during training.
Therefore, S2 only stored these lines in its memory.
However, S1 can also build net fragments for data not seen during training.

The following videos show the behaviour of the model if the line is rotated around the origin.


.. warning::
   If the videos are not shown, the encoding is not supported by your browser.
   Please try a Chromium-based browser (e.g. Google Chrome, Brave Browser, etc.).

**Fixed Threshold**


.. video:: ../_static/results/final_results/threshold/normal.mp4
   :width: 450

**Bernoulli Sampling**

.. video:: ../_static/results/final_results/bernoulli/normal.mp4
   :width: 450


Noise in the Input
~~~~~~~~~~~~~~~~~~

The following videos show the behaviour of the model if each *input* pixel is flipped with a probability of 0.005.
The model is not able to filter all noise. A reason is that each filter activates quite strongly to noise in the input data,
leading to many active neurons at the same position. This noise roughly correspond to the same pattern that can
be observed at line endings. Therefore, the noise receives lateral support and is not properly suppressed by inhibitory signals.

**Fixed Threshold**

.. video:: ../_static/results/final_results/threshold/005_noise.mp4
   :width: 450

**Bernoulli Sampling**

.. video:: ../_static/results/final_results/bernoulli/005_noise.mp4
   :width: 450


The same experiment is repeated, but this time the noise is added to each *feature channel* after the filter has been applied.
Thus, the same amount of noise is added, but not at the same position per channel.
As it can be observed, the model is able to filter the noise much better.
The reason is that the noise is not concentrated at the same position per channel, but is distributed over the whole image.
Therefore, the noise does not receive enough lateral support and is supressed.

**Fixed Threshold**

.. video:: ../_static/results/final_results/threshold/005_noise_per_channel.mp4
   :width: 450

**Bernoulli Sampling**

.. video:: ../_static/results/final_results/bernoulli/005_noise_per_channel.mp4
   :width: 450


Interrupted Line
~~~~~~~~~~~~~~~~

The following videos show the behaviour of the model if the line is interrupted in the middle.
Due to the lateral support, the model is able to reconstruct the line if up to 8 pixels are missing.
This is quite remarkable, as the model has never seen such a line during training and the lateral support range
is limited to 11 pixels.


**Fixed Threshold** for 5 missing pixels

.. video:: ../_static/results/final_results/threshold/5_black_pixels.mp4
   :width: 450

**Bernoulli Sampling**  for 5 missing pixels

.. video:: ../_static/results/final_results/bernoulli/5_black_pixels.mp4
   :width: 450

**Fixed Threshold** for 8 missing pixels

.. video:: ../_static/results/final_results/threshold/8_black_pixels.mp4
   :width: 450

**Bernoulli Sampling**  for 8 missing pixels

.. video:: ../_static/results/final_results/bernoulli/8_black_pixels.mp4
   :width: 450

**Fixed Threshold** for 10 missing pixels

.. video:: ../_static/results/final_results/threshold/10_black_pixels.mp4
   :width: 450

**Bernoulli Sampling**  for 10 missing pixels

.. video:: ../_static/results/final_results/bernoulli/10_black_pixels.mp4
   :width: 450