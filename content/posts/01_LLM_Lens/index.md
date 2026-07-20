---
title: "Geometric Lens: Verbalizing Hidden Representations of an LLM From First Principles"
author: Chunwei Ma
date: 2026-07-19T03:10:00-04:00
draft: false
math: true
showToc: true     # <-- This activates the Table of Contents layout
TocOpen: false    # <-- Optional: set to true if you want the ToC expanded by default
weight: 1  # <--- Pins this post to the absolute top
cover:
    image: "fig_lens/LLM_Lenses.png" # Path relative to the post folder
    alt: "Geometric Lens; Jacobian Lens; Logit Lens; Patchscopes"
    caption: "Comparison of LLM Lenses."
    relative: true # Tells PaperMod the path is relative to this post bundle
    hiddenInSingle: true # show as the cover, but not on top of the page    
---

[*This is an introductory blog for the new paper [Laguerre Geometry for Interpreting Large Language Models](https://arxiv.org/abs/2607.10578) and the GitHub repository [Geometric Lens](https://github.com/horsepurve/Geometric-Lens).*]

## LLM Lens: What does an internal vector mean?

Anthropic's recent paper on the "J-Lens" (Jacobian Lens) has revived interest in reading the "thoughts" inside Large Language Models. The idea of placing a “lens” at an LLM's hidden layers isn't new. It dates back to the Logit Lens, and has since evolved into a family of variants, including Tuned Lens and Patchscopes. But as we build new lenses, we keep hitting the same wall: what are we actually reading? How do we know that what we read is "correct"? Is there even a ground truth for the "meaning" of an internal state? These questions remain largely unanswered.

Like any deep neural network, an LLM has many layers, and each layer produces a high-dimensional vector (also called the residual stream). The vector after the final layer gets turned into vocabulary logits by the linear unembedding layer — so we do know the "meaning" (e.g., the top-1 predicted token) of that final vector. But what about the internal vectors? To answer this, let's first look at how existing methods try to decode them (Figure 1).

{{< figure 
    src="fig_lens/LLM_Lenses.png" 
    alt="LLM Lenses architectural overview" 
    link="fig_lens//LLM_Lenses.png" 
    target="_blank"
    class="center"
    title="Figure 1" 
    width="100%"
    caption="Illustration of Logit Lens, Tuned Lens, Patchscopes, Jacobian Lens, and Geometric Lens (ours). [View Full Size](fig_lens/)" 
>}}

* **Logit Lens** directly "moves" the hidden vector to the final representation space, generating the vocabulary logits straight away. This gives a coarse decoding, because the representation drifts as it passes through layers, so an early-layer vector doesn't actually live in the final layer's space.
* **Tuned Lens** improves on this by learning a transformation (an affine map) from each layer into the final layer, instead of assuming they already match. The catch: it needs a lot of training data (e.g., the Pile validation set) to learn this mapping for every single layer.
* **Patchscopes** takes a different approach entirely. It feeds the hidden vector into a separate model, together with a leading prompt (e.g., "The multi-tokens present here are..."), and treats the next token that model produces as the decoding. This means the result depends heavily on which model you use, how you word the prompt, and how exactly the patching is done.
* **Jacobian Lens** replaces the learned transformation with an averaged Jacobian matrix instead. This makes it "training-free" in the sense that it doesn't need gradient-based learning — but it still needs to be "fit" on a corpus of prompts (100–1,000 of them), so the result still depends on which corpus you chose. As a concrete data point, fitting it on 100 prompts took 51 minutes on a single GPU.

Notice the pattern: every one of these lenses is heuristic. None of them has a principled answer to "what is the ground-truth meaning of this hidden vector?" Instead, each one produces its decoding relative to something external — a training set, a second model and its prompt, or a prompt corpus. That means the "meaning" they output isn't unique. And if two different lenses disagree on what a hidden vector means, which one — if either — is actually correct?

To answer that question in a deterministic, principled way, this article defends the following claim: **the ground-truth meaning of a hidden vector should come from the model's own geometry, not from anything external to it**. Concretely, a network with piecewise-linear activation functions carves up its input space into many flat, linear "pieces," and computes something different (though still linear) within each piece. The claim is that a vector's true meaning is defined by which piece it falls into, not by training a separate transformation or feeding it into a second model.

To see why these two seemingly unrelated ideas — decoding hidden states, and how networks carve up space — are actually deeply connected, we first need to take a brief detour into how deep neural networks partition high-dimensional space.

## Classical finding: piecewise-linear regions of a deep neural network

{{< figure 
    src="fig_piecewise/piecewiselinear.jpg" 
    alt="The exact piecewise-linear space partitioning of a simple neural network" 
    link="fig_piecewise/piecewiselinear.jpg" 
    target="_blank"
    class="center"
    title="Figure 2" 
    width="50%"
    caption="Piecewise-linear space partitioning induced by a one-layer ReLU network with 3 classes. [View Full Size](fig_piecewise/)" 
>}}

Let's start with a well-known fact about ReLU networks. Take a tiny one-layer network with 12 hidden neurons as shown in Figure 2, used to classify points into 3 classes: input (2D) → hidden (12D) → output (3D).

* **Hidden layer (black lines)**: Each neuron in the hidden layer partitions the input space in half with a straight line. The 12 neurons cut independently of one another — when two of these lines cross, they don't interact with each other — so together they slice the 2D input plane into many small polygonal regions. This is a classic setting of hyperplane arrangement. 
* **Classification layer (red lines)**:  The output layer then draws a decision boundary between each pair of classes. (Technically, this forms what's called a Power Diagram with 3 sites in a 12-dimensional space.) Think of a red line simply as wherever a decision boundary between two classes happens to cross one of the small regions from the previous step. Each region either remains the same or gets cut further into 2 to 3 even smaller pieces.

Finally, every one of these small regions gets a color — the class it belongs to. If a new data point lands in a green region, we know it belongs to the green class.

## Piecewise-linear regions of an LLM

{{< figure 
    src="fig_llmtrees/LLM_static_trees.png" 
    alt="The static trees induced by an LLM" 
    link="fig_llmtrees/LLM_static_trees.png" 
    target="_blank"
    class="center"
    title="Figure 3" 
    width="60%"
    caption="Piecewise-linear space subdivision at each layer of an LLM. A region in layer l corresponds to many regions in its lower layer (l−1), forming a tree structure. The input 'is' falls into the bottom-most region of '\_a' in the embedding space, causing the next prediction to be '\_a'. [View Full Size](fig_llmtrees/)" 
>}}

Does this same picture apply to an LLM? Yes — with a couple of simplifications. If we restrict ourselves to a single input token producing a single output token, and assume ReLU activations (setting aside positional encoding and normalization for now), the network computes exactly this kind of piecewise-linear function. The piecewise-linear part of a layer consists of the whole MLP block, plus a portion of the MHA (Multi-Head Attention) block — its own-token stream. This is the intrinsic, static geometric structure of an LLM.

Put together, this creates a tree structure across layers. The top region of a class (e.g., the token “_a”) is a linear region, and it corresponds to many smaller piecewise-linear regions in its previous layer, so on and so forth. Keep going all the way back to the embedding layer, and you get one tree per output token: rooted at the token itself, branching into finer and finer regions the closer you get to the input. 

In Figure 3, the embedding vector for the input token "is" happens to fall inside one of the finest regions belonging to the tree rooted at token "\_a" — which is exactly why the model's top-1 prediction for "is" is "\_a". 

## What about attention? Cross-token creates jumps between trees

{{< figure 
    src="fig_transport/trasportation.png" 
    alt="The static trees induced by an LLM" 
    link="fig_transport/trasportation.png" 
    target="_blank"
    class="center"
    title="Figure 4" 
    width="70%"
    caption="A single input token flows within a static tree. When multiple input tokens exist, the inter-region transports move hidden vectors across trees. [View Full Size](fig_transport/)" 
>}}

First, suppose there's only one token, "is", as the input. As we saw, it falls into a linear region governed by the tree of "\_a", so it simply flows inside that static tree as it moves through the layers, ending in "\_a" as the output.

Now add a prefix: "The capital of France is". A token's hidden vector is shaped by two different contributions — how much it attends to itself (its own-token stream), and how much it attends to other tokens in the sequence (its cross-token stream, carrying information from "The", "capital", "of", "France"). It's specifically this second contribution — attention to other tokens — that causes what we call an inter-region transport: it can shove the hidden vector out of its current region and into a different one. That new region might still belong to the same tree, or it might belong to an entirely different tree (see Figure 4). Once a transport has happened, the vector resumes flowing inside whatever static tree it landed in — until the next transport moves it again.

In short, a hidden vector inside an LLM only ever does two things:

1. **Flow** inside a static tree, moving through piecewise-linear regions layer by layer.
2. **Jump** between different trees, via cross-token attention.

## The solution

So, for a hidden vector at some layer, how do we figure out its meaning? The answer is now clear: find the label of the linear region it falls into. And we already know how to do that — as long as there's no inter-region transport, the vector's flow is confined to a single static tree, and every region along that flow shares one label: the token sitting at that tree's root.

This gives us an answer to the question we started with:

> The ground-truth meaning of a hidden state in an LLM is the label of the piecewise-linear region containing that vector.

And it gives us a concrete recipe for reading it out: set every transport vector to zero, and let the hidden vector flow to the final layer on its own. In practice, this means removing all cross-token attention streams during the forward pass — but keeping the own-token streams intact, since removing those too would break the model's geometric structure entirely. We call this method the **Geometric Lens**.

## How do we evaluate an LLM lens?

With this new lens — the one we believe is the true ground-truth lens — a natural question follows: how do we actually confirm that it's the ground truth? This turns out to be a genuinely hard problem. Confirming any lens is ground truth requires already knowing the ground truth, which is circular — you'd need at least two independent ways of arriving at the ground truth, and then check that they agree. In practice, most lens papers sidestep this and rely on indirect evaluations instead, such as attribution extraction.

In this article, we offer two new angles for evaluating and comparing different lenses.

## Joint visualization of LLM decision regions and reasoning trajectories

{{< figure 
    src="fig_probs/prob_curves.jpg" 
    alt="The static trees induced by an LLM" 
    link="fig_probs/prob_curves.jpg" 
    target="_blank"
    class="center"
    title="Figure 5" 
    width="60%"
    caption="Log rank (left column) and probability (right column) of the two tokens \_Paris (solid lines) and Paris (dotted lines), elicited by three lenses: Logit Lens, Patchscopes, and Geometric Lens. Geometric Lens is the only lens that detects both the factually correct token Paris and the switching phase between \_Paris and Paris. [View Full Size](fig_probs/)" 
>}}

Existing lenses operate under the assumption that every intermediary representation should ideally encode the same concept as the final layer. Tuned Lens, for instance, minimizes the KL divergence between the probability distribution produced by the lens at each intermediary layer and the final distribution produced by the frozen model. We challenge this assumption and argue instead that a “true” hidden representation, whatever it is verbalized, intentionally carries diverse, evolving meanings — and that this diversity can itself be leveraged usefully.

{{< figure 
    src="geometric_lens_paris/geometric_lens_vs_jacobian_lens_paris_text.jpg" 
    alt="The static trees induced by an LLM" 
    link="geometric_lens_paris/geometric_lens_vs_jacobian_lens_paris_text.jpg" 
    target="_blank"
    class="center"
    title="Figure 6" 
    width="100%"
    caption="The token revealed by different lenses at each layer for the prompt: '*The capital of France is*'. Four methods generate different token trajectories along the layers." 
>}}
{{< figure 
    src="geometric_lens_paris/geometric_lens_vs_jacobian_lens_paris.jpg" 
    alt="The static trees induced by an LLM" 
    link="geometric_lens_paris/geometric_lens_vs_jacobian_lens_paris.jpg" 
    target="_blank"
    class="center"
    title="Figure 7" 
    width="90%"
    caption="Joint visualization of decision regions and reasoning trajectories produced by four lenses for the prompt '*The capital of France is*'. [View Full Size](geometric_lens_paris/)" 
>}}

For a token appearing in a lens's output trajectory, we distinguish factual correctness from grammatical correctness. For the prompt "The capital of France is", for example, both "\_Paris" and "Paris" are factually correct, but only "\_Paris" (with a leading space) is grammatically correct. We track both tokens across all layers for all three lenses: Logit Lens, Patchscopes, and Geometric Lens. Results are shown in Figures 5 and 6, where an interesting trend emerges — one visible only through Geometric Lens: in the earlier layers, the ranks of "\_Paris" and "Paris" rise at a similar rate; in the middle layers, "Paris" takes priority, reaching the top-1 rank around layers 20–25; in the final layers, the rank of "Paris" drops sharply, and "\_Paris" emerges as the sole correct token. There is a pronounced peak in "Paris"'s probability between layers 20 and 25, a peak absent for both Logit Lens and Patchscopes. We further jointly visualize the trajectories and Laguerre cells in Figure 7. This 2D visualization makes clear that the three lenses follow markedly different trajectories: Patchscopes takes the shortest path of the three; Logit Lens passes through fewer cells; and Geometric Lens traces the most complex path, passing through the greatest number of regions. Notably, for this model (Phi-2) and prompt, Geometric Lens is the only lens whose trajectory starts from generic tokens ("\_a" and "\_the"), passes through the factually correct token "Paris", and finally settles in the "\_Paris" cell — consistent with the intuition that an LLM gradually refines its answer across layers.

This suggests that the LLM here first retrieves all factually correct candidates in its earlier layers, and only later resolves both factual and grammatical correctness. Geometric Lens is the only lens that reveals this mechanism.

Caveats: It is important to note that, this token-switching phenomenon is not observed in all model-prompt pairs, but when there exists this phenomenon, the lens should be able to detect it.

## Geometric Lens Reveals a Model's Internal Reasoning under In-Context Interference

In this section, we focus on prompts of the form: "*You are in a fictional world where Marseille and Lyon have swapped their names. The Louvre Museum is located in the city of*", where a prefix mentioning two irrelevant cities A and B precedes a fact about a third city, C. We find that most models answer with the misleading city A or B more than 90% of the time. The question we investigate is which lens, if any, is able to show whether the model has ever thought about the correct answer in a certain layer. 

{{< figure 
    src="geometric_lens_lyon/geometric_lens_vs_jacobian_lens_lyon_text.jpg" 
    alt="The static trees induced by an LLM" 
    link="geometric_lens_lyon/geometric_lens_vs_jacobian_lens_lyon_text.jpg" 
    target="_blank"
    class="center"
    title="Figure 8" 
    width="100%"
    caption="The token revealed by different lenses at each layer for the prompt: '*You are in a fictional world where Marseille and Lyon have swapped their names. The Louvre Museum is located in the city of*'. Four methods generate different token trajectories along the layers." 
>}}
{{< figure 
    src="geometric_lens_lyon/geometric_lens_vs_jacobian_lens_lyon.jpg" 
    alt="The static trees induced by an LLM" 
    link="geometric_lens_lyon/geometric_lens_vs_jacobian_lens_lyon.jpg" 
    target="_blank"
    class="center"
    title="Figure 9" 
    width="90%"
    caption="Joint visualization of decision regions and reasoning trajectories produced by four lenses for the prompt '*You are in a fictional world where Marseille and Lyon have swapped their names. The Louvre Museum is located in the city of*'. [View Full Size](geometric_lens_lyon/)" 
>}}

We investigate the token-switching phenomenon and the illustration is shown in Figures 9. Among the three lenses, Geometric Lens is the only one for which the correct token “_Paris” ever attains the top-1 rank at a middle layer. The 2D visualization further shows that Geometric Lens is the only lens whose trajectory passes through the “_Paris” cell before finally settling in the  “_Lyon” cell. These findings suggest that, although the final output token is always the same, different lenses can elicit markedly different trajectories at middle layers — and only the correct lens reveals the model's true, and often complex, internal reasoning process.

Caveats: While for this model (Phi-2) and this specific prompt, Geometric Lens successfully recovers the correct token by using the output at a certain middle layer, there is no guarantee that Geometric Lens always elicits the correct token for all model-prompt pairs.

## Conclusion

Lenses have long remained heuristic, lacking a clear theory of what the "ground truth" for a hidden representation should be. In this article, we argue that this ground-truth label should instead be derived from the piecewise-linear subdivision of the model itself. Our theory assigns a label to each response region, thereby making it possible to label every hidden vector contained within it. This labeling motivates a new lens, the Geometric Lens, which we validate empirically both in regular forward passes and under in-context interference.

Our goal is to establish a ground truth for hidden representations inside an LLM, and to show that the Geometric Lens surfaces phenomena that other lenses miss — but we don't expect it to replace those other lenses. Even if a "ground-truth" lens exists, it won't necessarily be the most useful one for every downstream task. What we're offering instead is a principled answer to "what is the ground-truth meaning of a hidden vector?", together with a geometric framework for interpreting an LLM as a decomposition into piecewise-linear regions and inter-region transports. We hope this work makes the case that thinking about ground truth for a hidden vector matters — so that we know what any given lens can and cannot tell us.

For more details, read the [full story](https://arxiv.org/abs/2607.10578), or try out the [code](https://github.com/horsepurve/Geometric-Lens). 
