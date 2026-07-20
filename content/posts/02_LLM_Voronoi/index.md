---
title: 'Large Language Models Are Secretly Voronoi Diagrams'
author: Chunwei Ma
date: '2026-07-19T16:56:51-04:00'
draft: false
math: true
showToc: true     # <-- This activates the Table of Contents layout
TocOpen: false    # <-- Optional: set to true if you want the ToC expanded by default
weight: 2  # <--- Pins this post to the absolute top
cover:
    image: "figs/intro.jpg" # Path relative to the post folder
    alt: "Redefine concepts in LLM using Laguerre Geometry"
    caption: "LLM's concept geometry."
    relative: true # Tells PaperMod the path is relative to this post bundle
    hiddenInSingle: true # show as the cover, but not on top of the page    
---

[*This is an introductory blog for the new paper [Laguerre Geometry for Interpreting Large Language Models](https://arxiv.org/abs/2607.10578) and the GitHub repository [Geometric Lens](https://github.com/horsepurve/Geometric-Lens).*]

Large Language Models (LLMs) have achieved remarkable breakthroughs not only in general question-answering and conversation, but also in coding, mathematical reasoning, multimodal (image, audio, and video) generation, and scientific discovery. Yet, although every atomic internal computation within an LLM is ontologically well understood, the reason these computations, when composed, give rise to a high degree of intelligence remains largely epistemologically opaque—warranting deeper scientific inquiry.

A typical LLM is built on decoder-only transformers and trained autoregressively, with the model optimized to correctly predict the next token given a preceding piece of text. How this simple training objective gives rise to emergent capabilities such as comprehension and reasoning remains one of the central mysteries of LLMs. In moving from next-token probability distributions to emergent capabilities, the first natural question is how concepts are organized within an LLM's internal representations.

To answer this question, an intensively studied idea is the linear representation hypothesis (LRH)—the informal hypothesis that semantic concepts are represented linearly in the representation spaces of LLMs.
LRH has been widely substantiated across various contexts, including linear probes, vector arithmetic, intervention and activation steering, and circuit tracing.
However, four questions remain unresolved under this hypothesis: 

1. What is the definition of a concept?
2. How is a concept represented, and in which representation space?
3. What concrete geometric structure carries it?
4. Why should this structure be linear, and to what extent does linearity hold?

In this paper, we categorize LRH into two branches according to the representation space in which it is applied:
1. **Token classification layer**—the final linear projection layer of an LLM, which serves as the classification head and is trained with a cross-entropy loss, as in a standard deep neural network. A concept here is typically defined as a contrastive direction, a whitened unembedding vector, or the final hidden vector of a token. Most existing geometric analyses of LLMs focus on this layer.
2. **Intermediary layers** following each transformer block. LRH is the hypothesis underlying sparse autoencoders (SAEs), where high-level concepts are assumed to be linearly separable by latent units. A concept in this setting is defined as a latent direction that activates in response to a specific semantic meaning.

Some recent works challenge this vector/direction-based definition of concept in favor of clusters or regions. For instance, they model concepts as mixtures of Gaussian distributions, offering an alternative to SAEs. However, in these approaches a "region" remains an empirically defined cluster of activations, lacking an exact geometric structure.

From a reductionist standpoint, understanding the high-level phenomena of a system requires first understanding its most elemental constituent pieces. Prior work has shown that a feedforward neural network (FNN) with piecewise-linear activations partitions its input space into numerous piecewise linear regions; since FNNs account for the majority of an LLM's parameters, this same decomposition exists within LLMs as well. However, it remains unclear how this underlying geometry connects to the high-level semantic meanings encoded by an LLM.

In this paper, we propose a geometric framework that unifies low-level geometric decomposition with high-level semantic organization, connected via two complementary passes: top-down, we identify the semanticity-carrying regions in the top unembedding layer and progressively subdivide the space in lower layers; bottom-up, we track the region a prompt visits as it propagates from the input embedding layer to upper layers, assigning a meaning to each intermediate vector. In doing so, we move beyond hypothesis to concrete geometric objects that naturally exist within any transformer-based LLM. This instantiation of LRH allows us to precisely define both concepts and linearity in the final and intermediary representation spaces.

{{< figure 
    src="figs/laguerre.png" 
    alt="Laguerre Geometry Demonstration" 
    link="fig_lens/laguerre.png" 
    target="_blank"
    class="center"
    title="Figure 1" 
    width="50%"
    caption="A demonstration of all possible conditions of two oriented hyperspheres in Laguerre Geometry." 
>}}

Most existing geometric frameworks for LLM interpretability represent a concept via its unembedding vector—that is, as a single high-dimensional point. Laguerre Geometry extends this point to a high-dimensional hypersphere, admitting a much richer set of oriented contact conditions between two or more hyperspheres (e.g., disjoint, tangent, intersecting, contained, and concentric). In this paper, we ask: can extending concept points (vectors, directions) to concept hyperspheres reveal richer semantic knowledge embedded in a trained LLM? To answer this, we make the following contributions:

{{< figure 
    src="figs/power_diagram.png" 
    alt="Power Diagram Demonstration" 
    link="fig_lens/power_diagram.png" 
    target="_blank"
    class="center"
    title="Figure 2" 
    width="30%"
    caption=" An example Laguerre-Voronoi Diagram in 2D. In this work, a concept under LRH is defined as an entire cell, rather than a single point, vector, or direction." 
>}}

* We show that the embedding layer of an LLM induces a Laguerre-Voronoi Diagram (LVD) in the representation space. This diagram lets us strictly define concepts, linearity, and the relationships between them as concrete geometric objects. Laguerre Geometry also provides a powerful tool for modeling conceptual hierarchy.
* Moving from the "outer" embedding layer to the "inner" transformer blocks, we develop Geometric Lens, which uncovers the Laguerre region in which each hidden vector resides. We further develop Laguerre Autoencoder, a 2D visualization method that jointly renders the LLM's hidden representations alongside their corresponding LVD regions.
* We construct datasets, evaluation metrics, and experiments to assess linearity and hierarchy under different hypotheses, and to evaluate hidden-state elucidation across different lenses. We further show that our geometric framework effectively mitigates in-context interference.

