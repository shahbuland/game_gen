# Diffusion Distillation

This requires two steps: reflow, and LDD
reflow will come first. This will turn a regular v prediction model into a rectified flow model. From there,
we can distill to a one-step model. Models that are already rectified flow models don't need the extra step.