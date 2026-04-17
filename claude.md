---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.
I am inputting this information to help you understand the project better and to ensure that your contributions align with our coding standards and project goals. Please read through the following guidelines carefully: 


Track 1: DQN
Classical value-based deep reinforcement learning from pixels
Canonical paper: Human-level control through deep reinforcement learning (Mnih et al., 2015)
Reproduction target: Train DQN on ALE Breakout or ALE Pong through the Gymnasium or ALE interface.
Adaptation target: Move the method to MiniGrid MemoryEnv, a partially observed environment where the agent
must remember an object seen early in the episode and act on that information later.
Required method change: Add explicit memory beyond a standard feed-forward network, for example recurrent DQN,
gated state, or another well-justified memory mechanism.
Minimum success criteria
• Show that the baseline DQN learns a reasonable policy on Pong.
• Evaluate the unmodified feed-forward version on MemoryEnv and document where it fails or plateaus.
• Implement a memory-augmented variant and compare it against the baseline on the new environment.
• Report at least 3 seeds and compare both learning behavior and final performance.
Official links
• DQN paper (Nature)
• ALE documentation - official Atari environment family
• ALE Breakout - recommended reproduction option
• ALE Pong - alternate reproduction option
• MiniGrid environment index
• MiniGrid MemoryEnv - adaptation target


Requirements
• Use the official environment families linked in this document unless the instructor approves a close substitute in
advance.
• Separate the project into two phases: (1) canonical reproduction and (2) adaptation to a new environment.
• The adaptation must include at least one non-trivial algorithmic or architectural change. Pure hyperparameter
tuning, reward rescaling alone, or switching libraries is not enough.
• Run at least 3 random seeds for the main reported results.
• Track training reward or return and at least one task-specific metric such as success rate, collision count, or win
rate.
• Save enough information to reproduce the final plots: code version, configs, seed values, and the hyperparameters
used for the final runs.

Deliverables
• A code repository with a README and exact setup instructions.
• A compact appendix or table listing the final hyperparameters, environment versions, and compute budget.

Minimum evaluation evidence
• One plot or table for the canonical reproduction task.
• One direct comparison showing how the unmodified method behaves on the new environment.
• One comparison showing the adapted method on that same environment.
• At least one ablation isolating the main design change.

