.. Lateral Connections documentation master file, created by
   sphinx-quickstart on Mon May 22 17:18:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Self-Organisation in a Biologically Inspired Learning Framework Based on Bernoulli Neurons
==========================================================================================

Welcome to the online showcase of my master's thesis, titled "Self-Organisation in a Biologically Inspired Learning Framework Based on Bernoulli Neurons." This exploration delves into the crossroads of neuroscience and machine learning, focusing on the fascinating phenomenon of self-organisation within an novel learning framework.
Please find the abstract, the full thesis, and videos presenting the results below.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   results/final_results



.. warning::
    This is a pre-version of the thesis. A final version will be uploaded soon.


**Download the** :download:`thesis <_static/results/msc_thesis-v0-9-3.pdf>` **as PDF.**


Abstract
--------
In the past decade, deep learning has established itself as state-of-the-art technology in various automatic image analysis tasks.
Despite impressive results, this technology has several limitations, notably its limited robustness to noise, constrained transformation invariance during object recognition and reliance on a substantial amount of training data.
Conversely, the human brain does not suffer from these limitations due to its non-sequential processing of extracted image features and its ability to perceive visual scenes holistically, i.e. interpret it as more than the sum of its part, as outlined by Gestalt psychology.
This capability stems from the brain's ability to establish internal consistency between each connected cell pair through self-organisation and localised learning, i.e. a consensus is achieved across all features through mutual cell support. This mechanism solves the problem of ``early commitment'' inherent in deep networks as they rely on a global error correction algorithm to establish consistency at a single point between prediction and teaching signal.

This thesis builds upon these insights and proposes a novel image-processing framework inspired by the human brain's functionality.
Accordingly, a significant part of this thesis is devoted to identifying and interpreting neuroscientific findings.
These findings are analysed and translated into a computational framework, thereby linking each model component to the corresponding biological mechanism.

The framework consists of three components: The sensor system *S0*,  responsible for extracting low-level features from the images; the feature-building stage *S1*, which uses lateral (intra-layer) connections to form neuron groups, so-called net fragments,  fostering mutual support to stabilise known patterns; the prototype stage *S2*, which maps the formed net fragments to object prototypes using projection fibres and provides feedback to *S1*.
The iterative projection process between *S1* and *S2* lasts until consistency is achieved at every point in the network, i.e. until cells and synapses have reached a stable attractor state.

While prior research has demonstrated the efficiency of projection fibres, implementing net fragments still needs to be explored.
Consequently, this thesis analyses the implementation of this component in detail and discusses it by conducting experiments with a simple dataset based on straight lines.
The experimental findings demonstrate that lateral connections trained with Hebbian learning can facilitate cell support effectively.
The network exhibits significant robustness using cell support and can deactivate up to :math:`91.7\%` of unwanted cell activity triggered by noise signals. Furthermore, lateral support can restore discontinuous lines, demonstrating the network's ability to deal with occluded objects. With a range of lateral connections of :math:`11` pixels, interruptions of up to :math:`8` pixels can be reconstructed, and with additional feedback from *S2*, even interruptions of up to :math:`20` pixels can be restored. Improving the proposed framework can potentially reduce several weaknesses of conventional neural networks in the future and is considered a promising alternative research direction.


