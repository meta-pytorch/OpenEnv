# RFC: OpenEnv layering

**Status**: In Review
**Created**: 10/17/2025
**Authors**: @Darktex, @pankit-eng, @jspisak, @zkwentz
**RFC ID:** 000

## Summary
Before jumping into the actual concrete proposals, this RFC introduces how we are going to approach this problem space, what problems we want to solve with this project, and how we plan to prioritize and solve them systematically.

We recommend starting here to get a mental model of what's in here already, what is missing, and what's next.

## Goals
This project aims at standardizing environments for both training and evaluation. In the training space, this means also standardizing reward pipelines, while in the eval space this means helping with reproducibility where a model can be shipped with a complete set of agentic evals that can be easily run by others.

### The problem with abstraction boundaries
Ideally, we would draw a boundary between environments and everything else (orchestration, resource allocation, RPCs, etc). We will try to do this as much as possible, but we will have to create additional interfaces so that if folks want to cross this boundary, they can. This will likely be necessary for things like reward pipelines that call reward models (which will very likely need to RPC to GPU machines), as well as for agentic evals like Tau where the eval itself involve two agents interacting with one another (and sending many RPCs).

## Phases
We plan to build things incrementally, from inside out, adding and expanding only whenever necessary.

We will group development from now till version 1.0 into three phases.

In the **first phase** of this project, we will focus **exclusively** on the narrowest definition of environments, without even worrying about rewards nor evals. Instead, the focus in this phase (and in the RFCs you find in this directory) is going to be on:
1. Establishing a convention on what is an environment and where we draw the "environment" box.
2. Nailing our tools support
3. Landing the basics of _sandboxing_, _versioning_, _binary distribution_, _dependency management_.

We will conclude this phase with version 0.3.

In the **second phase** of this project, we will add rewards. Reward pipelines are crucial to get right since that is one of the main levels that ML engineers have to improve the model (much more than the general algorithm) so all the questions around versioning and deps that we tackled in the first phase will start being useful right away. We will introduce a way to RPC outside the Environment boundary: we expect this to require active discussion.

We will conclude this phase with version 0.6.

In the **third phase** of this project, we will add evals, starting from the simplest ones (simple evals that can be scored by a script), then extending to evals using LLM judges and then going into fully agentic evals.

We will conclude this phase with version 0.9.

Finally, we'll work on the finishing touches to get us from 0.9 to 1.0!

Throughout every phase, we will author concrete environments and we encourage everyone to do the same as developing together with use cases will give us line of sight into where we fall short and what needs to be changed/fixed.

Detailed success criteria and progress tracking for each version will be managed through GitHub milestones and issues.

## Our approach towards breaking changes
Until Version 1.0, we plan to move fast and will accept breaking changes if needed (they will be documented and posted into release notes, but we will not put soft deprecation in place until 1.0). We will have stronger guarantees in place after 1.0, to be determined down the road (probably together with the other "finishing touches" between 0.9 and 1.0...).

While APIs may not be 100% stable, we expect that any breaking changes should be relatively minor and can be fixed by LLM coders.

## Today: Version 0.1
We are unveiling this project early on to invite contributions from the get-go!

The official date for our 0.1 launch is October 22, 2025. If you are reading this sooner, spoiler alert!
