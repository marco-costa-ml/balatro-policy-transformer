# Policy Transformer

**Status:** This approach has been abandoned — see [Going Forward](#going-forward).

- This repository contains a "factorized hierarchical policy transformer." The model is capable of choosing from a decomposed action space of 100+ actions, with labels like SELECT_CARD_1, BUY_SHOP_ITEM_4, and SKIP_BLIND.

## Performance

- The model chose the correct action from an expert demonstration with 71% accuracy, and contained the correct decision within its top-3 actions 90% of the time, which was promising. However, in practice it performed very poorly, unable to pass the first blind with consistency.
- As it turns out, a decomposed action space leads to compounding errors, which is detrimental to long-run performance even over the course of a few actions.
- Furthermore, actions that are underrepresented (like SELL_JOKER_9) and that do not have strong corresponding signals will not be chosen by the model, despite using techniques like DAgger and logit adjustment.

## Literature

- Many of these issues have already been addressed by AlphaZero, a chess, shogi, and go agent developed by Google DeepMind in 2017. The solution for reasoning-heavy tasks is to leverage compute using simulators and value estimators.
- This is in stark contrast to AlphaStar, Google DeepMind's StarCraft II agent, which used a factorized hierarchical policy transformer and a decomposed action space.

## Going Forward

- In order to create an agent that is capable of playing an incomplete-information, stochastic, reasoning-heavy game, we need to use a very different approach.
- Instead of having a model make decisions directly, we train state estimators based on expert demonstration and simulate the results of high-level actions.
- Based on the results of several simulations and value estimates of each one, we can determine an "optimal" path and avoid the pitfalls of compounding error.
- The downside is that this requires a very hardware-conscious simulator, which does not exist for Balatro. I am working on this.
