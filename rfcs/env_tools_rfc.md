# [Proto RFC] Env & tools spec

Before this becomes an actual RFC, let me first write here so we can all review in google docs which is frankly easier to work with than Markdowns on github.

This proposal iterates on top of multiple RFCs (well... almost all of them). If we approve it, we will have to revise them accordingly. In this doc, let's just focus on the idea and then we will figure out how to wordsmith changes into the RFCs on Github as a second step (and let's be real, the one who's gonna do it is Claude lol).

The whole reason behind this proposal is finding the best way to integrate MCP in a way that provides unique value, but more in general, it's a good opportunity to take another look at everything from first principles. It's not bad to do this: this is why we are in a RFC stage!

## Our audience

We provide value to:

1. Env builders, by giving them more reach to be able to be used in multiple projects that would otherwise require adapters, thus lowering the cost of entry
2. Model builders, by giving them more inventory which is a proven path towards improved model performance.
3. The scientific community, by giving them a path to reproducibility of setups including tools/rewards/evals.
4. Infra engineers, by giving them a clear and stable contract that allows for separation of concerns so they can focus on optimizing the backend

## Our principles

Let's start from the very beginning: what are our non-negotiable principles that we strive to stick to?

1. **Minimize deltas across a project's lifecycle.** One of the barriers to adoption of this tech is the deltas you have:
   1. Across every phase in a project's lifecycle: Training → Evals → Deployment
   2. Human ↔ Agent divergence.

   Deltas hurt every ML project, but RL is particularly susceptible to them. We already know this, so we should provide a holistic solution for it, by design.

2. **We are hands-on.** We do not stop with providing a spec. We should not refrain from providing quality of life features and ready-made, importable code. These need not be part of the spec proper, but we provide them because we ultimately want to provide value (as per above). They will be optional.

3. **We are economically opinionated.** We do not refrain from having opinions about how we want our stuff to be used: a fully unopinionated project lacks a spine and delights none. However, it is not our place to pick winners and losers with respect to AI research as we work in a yet-to-be-crystallized space. So, we see differences in opinions in the research community (e.g. codeact vs traditional function calling) as opportunities to validate the flexibility of our system which should seamlessly support both. We should not obsess over this if we feel that a winner is clear and that taking an opinion can provide a ton of value (especially in more established areas like e.g. containers), but in general we should not do this often.

4. **Our design is LLM-friendly.** We know what LLMs like and don't like. We know what their limitations are (example: limited context window). We always think of these as we validate our designs. When there are tradeoffs between LLM-friendliness and infrastructure-friendliness, we evaluate tradeoffs holistically.

## Design

Ultimately, any abstraction is just a Venn diagram drawn over a set of components: naturally, where you draw your Venn diagram is somewhat arbitrary and you may indeed have multiple legit answers.

So, let us start by littering the floor with all the components without any groupings. I will use examples to ground us concretely at each step.

### Components

Ultimately, the components we have are the following:

1. There's gonna be tasks coming in with questions → these are purely data.
2. These tasks are landing in a stateful world. Note that this is true for both deployment as well as for training (going back to our principle: they need to stay close to each other).
3. These tasks are solved by the agent that interacts with this stateful world via some interface
4. This interface, and the state are made of software, so there's gonna be deps and binaries to take care of
5. Code execution and/or bash access are technically optional but will be there so often, that in practice we are always going to need some form of sandboxing

To drive this example home, let's look at a real task.

Let's say I'm giving you a database containing a list of employees and as a single task, I'm giving you an ongoing task to stay alive over the course of months and maintain this database as events happen. This means querying for information, but it can totally also mean mutations. People get hired, people leave, and these mutations need to be performed.

![Main components](components.png)


I call the initial snapshot of the database our **state**. I would like to zoom in on it:

1. While it is made of data, it's not part of the dataset which normally contains tasks. You can have many different tasks operate on the same database snapshot!
2. While you query and mutate it via tools (e.g. MCP), it's not part of the MCP spec itself which only deals with interfacing to it.
3. This snapshot is relevant while training. You are essentially simulating a real-world scenario. Note that it's critical that we have the ability to reset the db snapshot to its original state, but **crucially!!** the model absolutely cannot do that. This is the function of the `.reset()` method in the Gymnasium API. It's a simulation reset. The `.reset()` method is absolutely not a tool that the model is free to call! For example, if the model decides to drop every record from the DB, our reward system will penalize it and it will learn not to do it in the future. We would then reset the simulation state back to the beginning, and try again. If this were to be exposed to the model, we would have a huge discrepancy with Prod as the model would learn during training that every error is always recoverable and thus it will have no risk aversion.

This is actually something that existing libraries do not do well, because they often bundle it with data, or directly with tools but unfortunately it does not belong in either place. Furthermore, let's make a note of something: we need a way of switching between Sim mode (Training, Eval) and Deployment Mode (Prod). Let's put a pin on this for now, but keep reading...

#### FAQs

**What about stateless worlds?** What about them? They sure exist but they are not all that interesting so the honest answer is that any half-baked library can support them without much issue. Even if you are playing chess or you have a Python interpreter you have a state and we are talking about the most basic environments ever...

**Isn't this what ARE is built for?** Not exactly. ARE/SIMS is built for a special case of this where the state also advances independently on what the agent is doing (so, there is an event queue with events firing irrespective of what the agent does. Example: in a self-driving environment, other cars will always move if you decide to stand still). Furthermore, SIMS is further optimized for phone apps talking to one another directly (example: someone tells you on Whatsapp that they will send you an email to confirm something and then they do send it a few minutes later. Which means you have some connection between Whatsapp ↔ GMail). Given that we are just staying high level with the idea of a "state" here, this still fits.

### Proposed abstractions

I would just define an **"Environment"** as **Interface (through MCP) + State**.

* **MCP is our interface to/from agents.** Note that this means exposing every action as a tool, including actions that are not what you would normally call tool-based. For example: chess. You normally would not expose moves as a MCP because it's not a "tool", but it is possible to do it, and you get the benefits of MCP.
  * A benefit that is perhaps non-obvious (at least, it wasn't to me until Pierre Andrews pointed it out: discovering the action space is not trivial. MCP naturally solves this as it has a way to list_tools which can also be used to discover actions).
  * Every query and every mutation performed by the agent will come through MCP calls.
  * We reserve special methods – at a minimum `.reset()`, we need to think of what else – that the model won't have access to. We currently expose these using our HTTP client which is a different path from MCP. On the one hand this is fine as there is a clear separation of concerns since these cannot be called by the model, on the other hand I am irked by the lack of symmetry...

* **We have a State (or Snapshot, or whatever we want to call it)** which gives us a way of instantiating a particular state of the world that is understood as the starting point for the simulation.
  * This is definitely true for Training and Evaluation
  * Note: even when running in Prod mode, a state may still be useful. For example, it can contain OAuth tokens to let the agent identify on a user's behalf and authenticate. We will need to figure this out.

In terms of responsibilities, I would add here:
* Sandboxing (code & filesystem)
* Binary distribution and versioning

So our example becomes this:

![Proposed abstraction](env.png)

### People, not just agents

We need to battletest this idea by trying things out on environments meant for people.

Assume that we expose the following tools via MCP:
* Screen (which takes care of rendering whatever you are doing inside and exposing it as an image... like an actual PC screen)
* Keyboard
* Mouse

Then, can we build a Docker that e.g. you can remote desktop to and perform some actions on (e.g. play a videogame) and give the agent exactly the same interface as a person had?

## Convenience features

### Traits

I can see a few convenience tools that we should bundle in optional "packages" that maybe you can import as traits:

* Human-Computer interfaces (discussed above: screen + keyboard + mouse, but also steering wheel + pedals + dash + mirrors if you are driving a car etc).
* Bash and filesystem access
* Standard data science Python environment (pandas, pytorch, numpy, matplotlib, mypy etc)

I think having an ergonomical way of bundling these in code and making these importable would be nice, like:

```python
class CompetitiveCodingEnv(HumanComputerInterfaceable, BashAccess, PythonAccess):
    # Example using the above as mixins
    ...
```

### CodeAct and ToolCall

If our users write everything based on MCP, we can programmatically switch between tool calling and CodeAct style on the fly. Explanation:

1. **Tool calling-style.** This style enables a defined action-space (which is actually more in line to the Gymnasium philosophy), where each tool call is an action and the result of the call is shown back to the LLM as an observation.

2. **CodeAct-style.** In this style, there is no defined action space: LLMs just write code, and tools are exposed as simply Python functions, so with a single action a LLM can write a code block that can do several tool calls, have control flows around them, loops etc. This is better because it doesn't expose every output to the model's context window (which is very limited), using the Python interpreter's working memory instead (which is dirt cheap). One example of why this wins: say your tool call returns 1M results but you are only interested in results that contain a certain string. In tool calling-style, you need to overwhelm the LLM's context window with all 1M results, while in CodeAct you can write a Python block that will call the tool and filter the outputs, pasting to the LLM only after the filter. This is a huge deal because the single most limited resource we have is the model's context window.

This feature is a very big deal. CodeAct will inevitably win (it's just better for LLMs), but there are a ton of legacy agents based on tool-calling. I think that offering this seamless switching is gonna help a lot of people move over gradually and testing things throughout (also connected to our reproducibility goals).

### Tool discoverability

Another consequence of a model's short context window is that you cannot simply dump a super long list of a bajillion tools to it, lest you are left with no workable context window for your task!

There are several methods to mitigate this, but in practice they all kinda follow the same Data Structures 101 idea: you go from a list to a tree!

Anthropic proposes two simple approaches:

1. Make each server a directory, and each tool a file under that directory using descriptive names so the model can explore and figure out which ones to open to read the definition based on their name. This works if O(MCP Servers) is relatively small and if O(Tools) is large (i.e. few big servers)
2. Simply build a new tool, `find_tools` to go search for you.

Others have done the same thing, in my opinion with a worse (overcomplicated) design. This generally involves building some sort of Gateway abstraction which ultimately does the same two things as above (what else are you gonna do?).

* Docker MCP Gateway
* AWS AgentCore Gateway
* MCP Router

I would simply make `find_tools` one of the "quality of life" tools that we offer. We naturally have the directory-like interface as well which is also gonna look quite nice to LLMs.

```
coding_environment/  # reading this, the LLM is gonna expect to see compilers and whatnot in here
    compiler_mcp_server/  # Jury still out if we should have one or more MCP servers
        compile.py
        link.py
        benchmark.py
    browser_mcp_server/
        google_search.py
        navigate_to.py
        ...
```

#### FAQs

**Couldn't MCP servers extend to support a state and "put us out of business"?** The closest thing that MCP has to a state is its support of sessions and of authorization. While it's true that you could probably implement our state in a MCP server by wrangling sessions, it's not what they were made for. MCP was not made for training! All the stuff that we have been listing still applies even if we decide to implement this behind the scenes using MCP sessions.

## From environments to worlds

I don't think the env abstraction alone is enough to realize all the value we can be providing. At a minimum, I believe we need one more wrapper to ship data together with the environment: while it is true that environments can be reused, it's just as true that many environments are meant for specific datasets and thus it is natural for users to bundle them. We know this for a fact, because we also started to see pull requests that wanted to bundle datasets with the environment.

Now, if we are also looping on tasks, we should take a look at the whole flow. Let's do it in code, it's quite readable anyway:

```python
for task in tasks: # First loop is over a dataset
    obs = env.reset() # New task, clean simulation state
    while not task.done: # Second loop allows for followups, long-running sub-tasks "now do this" etc
        for role in (user, agent):
            if role == "agent":
                while True:  # inner loop is infinite Agentic loop: thinks, acts, observes until satisfied
                    # 3 stages conceptually, but in practice you just interleave 1 llm_call to 1 env
                    thought, action = llm_call(obs)  # one llm call does all of them.

                    if thought.done:
                        return thought.final_answer  # some put it as an action

                    obs = env.step(action)   # dump results into prompt,
                    # ready to be observed by next llm_call
```

While Forge can write a loop like the above, something that we can do in OpenEnv is simply provide the methods and abstractions, and let the caller call them however they please.

We need a good name for this abstraction, let's try "Worlds" for now.

**World = Data + Environment + Evals + Rewards**

Now, we can have methods like:

```python
task = world.next_task()  # samples next task from the dataloader/queue (if it's a user)

# worlds can also do agentic evals where two agents ping-pong against each other
world.run_evals()

world.rewards()
```

## What I am undecided on

* Should rewards be part of the environment or of the world?
* How do we handle binary deps that are only for the world but not for the env?
* How do we avoid creating dockers of dockers?
* Right now we layer things this way:
  * MCP Server + State → Env
  * Env + Data + Evals + Rewards → World
