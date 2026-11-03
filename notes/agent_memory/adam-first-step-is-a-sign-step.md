---
name: adam-first-step-is-a-sign-step
description: Adam's t=1 update equals -lr*g/(|g|+eps), so update-gradient cosine is never -1 even from random init with no weight decay
metadata:
  type: reference
---

At Adam's first step the bias corrections cancel exactly: `m_hat = g`, `sqrt(v_hat) = |g|`, so the update is `-lr * g / (|g| + eps)`, essentially `-lr * sign(g)`. The cosine between update and gradient is therefore `-||g||_1 / (sqrt(n) * ||g||_2)`, which is about -0.8 for iid Gaussian coordinates and was measured at -0.68 on the depth-5 ODT ReLU run. It reaches -1 only if every coordinate has equal magnitude.

Plain SGD with zero weight decay does give exactly -1. Two caveats that came up when checking this: SGD's weight decay is coupled and is *not* written into `param.grad`, so a nonzero decay wrecks the measured cosine; and updates recorded as `params_after - params_before` in float32 lose about two digits when the step is small, which showed as -0.99995 instead of -1 at `lr = 2e-3`.

Relevant to [[user-research-direction]] because it means Adam departs from the gradient direction immediately, before any mini-batch noise or momentum history accumulates. Tooling in [[relu-replayable-training]].
