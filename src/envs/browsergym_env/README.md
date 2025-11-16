
# BrowserGym Environment

BrowserGym is a unified framework for web-based agent tasks that provides access to multiple benchmarks under a single Gymnasium-compatible API. This integration brings the complete training-to-evaluation pipeline for web agents into OpenEnv.

## Why BrowserGym?

BrowserGym provides a complete pipeline for developing web agents: train on simple tasks, then evaluate on realistic websites.

**What are these benchmarks?**

- **MiniWoB++ (Training)**: 100+ synthetic web tasks like "click this button", "fill out this form", "select from dropdown". Each task is a simple webpage with a clear objective. Fast resets, randomized variations, dense rewards. Perfect for learning basic web navigation skills. **No external setup needed** - tasks run in isolated browser sessions.

- **WebArena (Evaluation)**: 812 tasks on real websites (e-commerce, forums, GitLab, Wikipedia). Tasks like "find the cheapest laptop and add to cart" or "create a merge request for bug #123". Multistep, requires reasoning, sparse rewards. Tests if your agent can handle actual websites. **Requires running 7 backend services** (shopping site, GitLab instance, etc.).

- **VisualWebArena**: Similar to WebArena but requires visual understanding - agents need to interpret images, identify UI elements visually, handle multimodal content.

- **WorkArena**: Enterprise software tasks (CRM, project management, business workflows). Tests automation on corporate-style applications.

**The training → evaluation pipeline:**
1. Train on MiniWoB (simple, controlled, fast iterations)
2. Evaluate on WebArena (complex, realistic, measures real-world capability)

**Key advantage**: You can start training immediately with MiniWoB. No need to set up infrastructure just to test if your code works.

## Quick Start - Training (MiniWoB)

### No Setup Required! 

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for MiniWoB training task
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",  # or "click-button", "click-dialog", etc.
    }
)

# Train your agent!
for episode in range(1000):
    result = env.reset()
    print(f"Goal: {result.observation.goal}")

    done = False
    while not done:
        # Your agent decides what to do
        action_str = agent.get_action(result.observation.text)
        action = BrowserGymAction(action_str=action_str)

        result = env.step(action)
        done = result.done

        print(f"Reward: {result.reward}")

env.close()
```

## Custom Tasks - Create Your Own Benchmarks

In addition to official BrowserGym benchmarks (MiniWoB, WebArena, etc.), you can create **custom tasks** for domain-specific training or prototyping.

### Why Custom Tasks?

**Official Benchmarks** (miniwob, webarena):
-  Established, well-tested tasks
-  Standardized evaluation
-  Community benchmarks
-  Fixed task set - can't add your own
-  Requires BrowserGym package installation
-  Must integrate with BrowserGym's registration system

**Custom Tasks**:
-  Create unlimited domain-specific tasks
-  No BrowserGym package needed
-  No registration complexity
-  Full control over HTML, rewards, termination
-  Rapid prototyping and iteration
-  Not standardized (for research/training only)

### Quick Start - Custom Tasks

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Use a custom task (no BrowserGym installation needed!)
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "custom",
        "BROWSERGYM_TASK_NAME": "copy-paste",  # or "copy-paste-multitab"
    }
)

# Train on your custom task
result = env.reset()
print(f"Goal: {result.observation.goal}")

action = BrowserGymAction(action_str="click('#source-text')")
result = env.step(action)
print(f"Reward: {result.reward}")

env.close()
```

### Available Custom Tasks

| Task Name | Description | Difficulty | Multi-Page |
|-----------|-------------|------------|------------|
| `copy-paste` | Copy text from one field to another | Easy | No |
| `copy-paste-multitab` | Copy text across two pages | Medium | Yes |

### Action Format Reference

Custom tasks support BrowserGym-style action strings:

- **Click**: `click('button')` or `click('#submit')` or `click('.classname')`
- **Fill**: `fill('input[name="username"]', 'john@example.com')`
- **Navigate**: `goto('https://example.com')` or `goto('file:///path/to/page.html')`
- **Press key**: `press('Enter')` or `press('Control+C')`
- **Scroll**: `scroll('down')` or `scroll('up')`
- **Custom JavaScript**: Any other string is executed as JavaScript in the browser context

**Examples:**
```python
# Click actions
BrowserGymAction(action_str="click('#submit-btn')")
BrowserGymAction(action_str="click('button.primary')")

# Fill forms
BrowserGymAction(action_str="fill('#email', 'user@example.com')")
BrowserGymAction(action_str="fill('input[name=\"password\"]', 'secret123')")

# Keyboard
BrowserGymAction(action_str="press('Tab')")
BrowserGymAction(action_str="press('Control+A')")

# Navigation
BrowserGymAction(action_str="goto('https://example.com')")

# JavaScript (for complex interactions)
BrowserGymAction(action_str="document.querySelector('#dropdown').value = 'option2'")
```

### Creating Custom Tasks

Custom tasks are defined in `server/custom/custom_tasks.py`. Each task needs:

1. **Task HTML** - Minimal HTML page(s) with your UI
2. **Python Task Class** - Defines behavior, rewards, termination
3. **Registration** - Add to task registry

**File Structure:**
```
server/custom/
 custom_models.py       # CustomGymAction, CustomGymObservation, CustomGymState
 custom_base.py         # Base class for custom environments
 custom_tasks.py        # Task registry and implementations
 tasks/                 # HTML files for tasks
     copy-paste.html
     copy-paste-source.html
     copy-paste-target.html
```

**Design Philosophy** (Following Official Benchmarks):

 **DO:**
- Keep HTML minimal and functional (like MiniWoB)
- Let agents figure out what to do from task description
- Use simple, clean styling
- Focus on task logic, not visual appeal

 **DON'T:**
- Add step-by-step instructions in HTML
- Use fancy animations or gradients
- Add visual hints or progress indicators
- Use emojis or decorative elements

**Example: Single-Page Task**

```python
# In server/custom/custom_tasks.py
from custom_base import CustomBrowserGymEnvironment

class MyCustomTask(CustomBrowserGymEnvironment):
    def _get_task_url(self) -> str:
        """Return path to your HTML file."""
        import os
        task_html = os.path.join(
            os.path.dirname(__file__),
            "tasks",
            "my-task.html"
        )
        return f"file://{task_html}"
    
    def _get_goal_description(self) -> str:
        """Return task instruction for the agent."""
        return "Click the submit button after filling the form"
    
    async def _extract_observation(self, page) -> dict:
        """Extract state from the page."""
        content = await page.content()
        form_valid = await page.evaluate(
            "document.querySelector('form')?.checkValidity() || false"
        )
        
        return {
            "text": content,  # Full HTML for agent
            "pruned_html": content[:1000],  # Truncated version
            "custom_data": {
                "form_valid": form_valid,
            }
        }
    
    def _calculate_reward(self, page_data, action, error=None) -> float:
        """Calculate reward based on page state."""
        if error:
            return -0.1  # Small penalty for errors
        
        custom_data = page_data.get("custom_data", {})
        if custom_data.get("form_valid"):
            return 1.0  # Success!
        
        return 0.0  # No progress
    
    def _check_done(self, page_data) -> bool:
        """Check if task is complete."""
        custom_data = page_data.get("custom_data", {})
        return custom_data.get("form_valid", False)

# Register your task in server/custom/custom_tasks.py
register_custom_task("my-task", MyCustomTask)
```

**Step-by-Step Registration:**
1. Create your task class in `server/custom/custom_tasks.py` (or import it)
2. Call `register_custom_task("task-name", YourTaskClass)` at the bottom of the file
3. Create HTML file(s) in `server/custom/tasks/` directory if needed
4. Use with `BROWSERGYM_TASK_NAME="task-name"`

**Example: Multi-Page Task**

```python
class MyMultiPageTask(CustomBrowserGymEnvironment):
    async def _extract_observation(self, page) -> dict:
        content = await page.content()
        current_url = page.url
        
        # Determine which page we're on
        if "page1" in current_url:
            data = await page.evaluate("getPage1Data()")
            return {
                "text": content,
                "custom_data": {"current_page": "page1", **data}
            }
        elif "page2" in current_url:
            data = await page.evaluate("getPage2Data()")
            return {
                "text": content,
                "custom_data": {"current_page": "page2", **data}
            }
        
        return {"text": content, "custom_data": {}}
    
    def _calculate_reward(self, page_data, action, error=None) -> float:
        """Reward for navigation and completion."""
        custom_data = page_data.get("custom_data", {})
        current_page = custom_data.get("current_page")
        
        # Reward for successfully navigating to page2
        if current_page == "page2" and "goto" in action.lower():
            return 0.3
        
        # Reward for task completion on page2
        if current_page == "page2" and custom_data.get("task_complete"):
            return 1.0
        
        return 0.0

register_custom_task("my-multitab-task", MyMultiPageTask)
```

### Custom Task HTML Guidelines

Follow official benchmark style (MiniWoB, WebArena):

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Task</title>
    <style>
        /* Minimal, functional styling only */
        body { font-family: Arial, sans-serif; padding: 20px; }
        input { padding: 5px; margin: 10px 0; }
        button { padding: 8px 20px; }
    </style>
</head>
<body>
    <!-- No instructions! Agent learns from goal description -->
    <input type="text" id="name" placeholder="Name">
    <input type="email" id="email" placeholder="Email">
    <button id="submit">Submit</button>
    
    <div id="success" style="display:none;">Success!</div>
    <div id="error" style="display:none;">Error: Please fill all fields.</div>
    
    <script>
        document.getElementById('submit').onclick = function() {
            var name = document.getElementById('name').value;
            var email = document.getElementById('email').value;
            
            if (name && email) {
                document.getElementById('success').style.display = 'block';
            } else {
                document.getElementById('error').style.display = 'block';
            }
        };
    </script>
</body>
</html>
```

**Key Principles:**
- No visual hints or progress bars
- No step-by-step instructions in HTML
- No emojis or decorative elements
- Simple, clean, functional UI
- Agent figures out task from goal description

### Custom vs Official Benchmarks

| Aspect | Official (miniwob, webarena) | Custom |
|--------|----------------------------|--------|
| **Installation** | Requires browsergym-{benchmark} | No packages needed |
| **Task Creation** | Fixed task set | Unlimited custom tasks |
| **Registration** | gym.make() system | Simple Python registry |
| **Browser Control** | BrowserGym internals | Playwright directly |
| **HTML Location** | BrowserGym package | Local server/custom/ directory |
| **Use Case** | Standardized evaluation | Rapid prototyping, domain-specific training |
| **Community** | Established benchmarks | Your own tasks |

### When to Use Custom Tasks

 **Use Custom Tasks For:**
- Rapid prototyping of new task ideas
- Domain-specific training (e.g., corporate workflows, specialized forms)
- Testing new agent architectures quickly
- Educational purposes
- Tasks not covered by official benchmarks

 **Use Official Benchmarks For:**
- Publishing research results
- Comparing with other papers
- Standardized evaluation
- Established task benchmarks

### Advanced: Custom Task Features

**Dynamic Task Generation:**
```python
class DynamicFormTask(CustomBrowserGymEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_fields = self.custom_params.get("num_fields", 3)
    
    def _get_task_url(self) -> str:
        # Generate HTML dynamically
        html = self._generate_form_html(self.num_fields)
        # Use data: URL
        return f"data:text/html,{html}"

# Use with custom parameters
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "custom",
    "BROWSERGYM_TASK_NAME": "dynamic-form",
    "num_fields": "5"  # Custom parameter
})
```

**State Persistence Across Pages:**
```python
class MultiPageWithState(CustomBrowserGymEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_state = {}  # Persistent state
    
    def _calculate_reward(self, page_data, action, error=None) -> float:
        # Access state from previous pages
        if self.task_state.get("collected_item"):
            return 1.0
        return 0.0
```

**See also:**
- `server/custom/README.md` - Detailed custom task documentation
- `server/custom/custom_tasks.py` - Example implementations
- `examples/browsergym_custom_example.py` - Usage examples

---

## Evaluation (WebArena)

#### MiniWoB++ Tasks (Training - 100+ tasks)

MiniWoB tasks are organized by difficulty and type. Here are the main categories:

**Click Tasks** (Basic interaction)
| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `click-test` | Click a single button |  Easy |
| `click-button` | Click button with specific text |  Easy |
| `click-button-sequence` | Click buttons in order |  Medium |
| `click-checkboxes` | Select specific checkboxes |  Medium |
| `click-checkboxes-soft` | Select checkboxes (multiple valid) |  Medium |
| `click-checkboxes-large` | Many checkboxes to select from |  Medium |
| `click-checkboxes-transfer` | Transfer learning variation |  Medium |
| `click-dialog` | Click correct button in dialog |  Easy |
| `click-dialog-2` | More complex dialog |  Medium |
| `click-link` | Click on a link |  Easy |
| `click-option` | Select from dropdown |  Medium |
| `click-pie` | Click on pie chart slice |  Medium |
| `click-scroll-list` | Click item in scrollable list |  Hard |
| `click-shades` | Click on specific color shade |  Medium |
| `click-shape` | Click on specific shape |  Medium |
| `click-tab` | Switch between tabs |  Medium |
| `click-tab-2` | More complex tab switching |  Hard |
| `click-widget` | Click on UI widget |  Medium |

**Text Entry Tasks** (Typing and forms)
| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `enter-text` | Type text into input field |  Easy |
| `enter-text-dynamic` | Dynamic text entry |  Medium |
| `enter-text-2` | Multiple text fields |  Medium |
| `enter-password` | Fill password field |  Easy |
| `enter-date` | Enter a date |  Medium |
| `enter-time` | Enter a time |  Medium |
| `login-user` | Complete login form |  Medium |
| `login-user-popup` | Login via popup |  Hard |

**Navigation Tasks** (Multi-step interaction)
| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `navigate-tree` | Navigate through tree structure |  Hard |
| `search-engine` | Use search interface |  Medium |
| `use-autocomplete` | Interact with autocomplete |  Hard |
| `book-flight` | Book a flight (complex form) |  Very Hard |
| `choose-date` | Pick date from calendar |  Hard |
| `choose-date-easy` | Simplified date picker |  Medium |
| `choose-date-medium` | Medium difficulty date picker |  Hard |
| `choose-list` | Select from long list |  Medium |

**Visual/Spatial Tasks** (Requires visual understanding)
| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `count-sides` | Count sides of shape |  Medium |
| `count-shape` | Count specific shapes |  Medium |
| `find-word` | Find word in text |  Medium |
| `focus-text` | Focus on text element |  Easy |
| `focus-text-2` | More complex focus task |  Medium |
| `grid-coordinate` | Click grid coordinate |  Medium |
| `guess-number` | Guess a number game |  Hard |
| `identify-shape` | Identify shape type |  Medium |
| `read-table` | Extract info from table |  Hard |
| `read-table-2` | More complex table reading |  Hard |

**Email/Social Tasks** (Realistic scenarios)
| Task Name | Description | Difficulty |
|-----------|-------------|------------|
| `email-inbox` | Manage email inbox |  Very Hard |
| `email-inbox-forward` | Forward emails |  Very Hard |
| `email-inbox-nl` | Natural language email task |  Very Hard |
| `email-inbox-star-reply` | Star and reply to emails |  Very Hard |
| `social-media` | Social media interaction |  Very Hard |
| `social-media-some` | Partial social media task |  Hard |

**Total:** 100+ tasks across all categories

**Usage:**
```python
# Easy task for quick testing
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "click-test"})

# Medium difficulty for training
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "click-checkboxes"})

# Hard task for evaluation
env = BrowserGymEnv(environment={"BROWSERGYM_TASK_NAME": "email-inbox"})
```

#### WebArena Tasks (Evaluation - 812 tasks)

WebArena tasks are organized by website and difficulty. Tasks are numbered 0-811.

**By Website:**
| Website | Task Count | Description | Example Tasks |
|---------|------------|-------------|---------------|
| Shopping | ~200 | E-commerce site | Search products, add to cart, checkout |
| Shopping Admin | ~150 | Admin panel | Manage products, orders, customers |
| Reddit | ~150 | Forum/social | Post, comment, search discussions |
| GitLab | ~200 | Code repository | Create issues, merge requests, review code |
| Wikipedia | ~100 | Knowledge base | Search, read, extract information |
| Map | ~12 | Location service | Find places, get directions |

**By Difficulty:**
| Difficulty | Task Count | Steps Required | Example |
|------------|------------|----------------|---------|
| Easy | ~200 | 1-5 steps | "Find the price of product X" |
| Medium | ~400 | 5-15 steps | "Add cheapest laptop to cart" |
| Hard | ~212 | 15+ steps | "Create merge request for bug fix" |

**Usage:**
```python
# Task 0 (usually easy)
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "webarena",
    "BROWSERGYM_TASK_NAME": "0",
    "SHOPPING": "http://your-server:7770",
    # ... other URLs
})

# Task 156 (GitLab merge request)
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "webarena",
    "BROWSERGYM_TASK_NAME": "156",
    # ... URLs
})
```

**Note:** WebArena tasks require the full backend infrastructure. See [WebArena setup guide](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

#### VisualWebArena Tasks (910 tasks)

Similar to WebArena but requires visual understanding. Tasks involve:
- Image-based reasoning
- Visual element identification
- Multimodal interaction (text + images)

#### WorkArena Tasks

Enterprise software automation tasks:
- CRM operations
- Project management
- Business workflows

**Full task lists:**
- [MiniWoB++ tasks](https://github.com/Farama-Foundation/miniwob-plusplus/tree/master/miniwob/environment)
- [WebArena tasks](https://github.com/web-arena-x/webarena/blob/main/config_files/)
- [BrowserGym documentation](https://github.com/ServiceNow/BrowserGym)

## Evaluation (WebArena)

### Prerequisites

WebArena requires setting up backend infrastructure. See the [WebArena documentation](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

### Usage

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for WebArena evaluation
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",  # Task ID
        # WebArena backend URLs (required)
        "SHOPPING": "http://your-server:7770",
        "SHOPPING_ADMIN": "http://your-server:7780/admin",
        "REDDIT": "http://your-server:9999",
        "GITLAB": "http://your-server:8023",
        "MAP": "http://your-server:3000",
        "WIKIPEDIA": "http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": "http://your-server:4399",
    }
)

# Evaluate your trained agent
result = env.reset()
while not result.done:
    action_str = agent.get_action(result.observation)
    action = BrowserGymAction(action_str=action_str)
    result = env.step(action)

print(f"Success: {result.reward}")
env.close()
```

## Building the Docker Image

### Prerequisites

1. **Base Image**: Build the OpenEnv base image first:

```bash
# From the OpenEnv repository root
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
```

### Build the BrowserGym Environment

```bash
# From the OpenEnv repository root
docker build -t browsergym-env:latest -f src/envs/browsergym_env/server/Dockerfile .
```

### Run the Server

#### For MiniWoB (Training):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest
```

#### For WebArena (Evaluation):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="webarena" \
  -e BROWSERGYM_TASK_NAME="0" \
  -e SHOPPING="http://your-server:7770" \
  -e SHOPPING_ADMIN="http://your-server:7780/admin" \
  -e REDDIT="http://your-server:9999" \
  -e GITLAB="http://your-server:8023" \
  -e MAP="http://your-server:3000" \
  -e WIKIPEDIA="http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  -e HOMEPAGE="http://your-server:4399" \
  browsergym-env:latest
```

## Environment Details

### Action

Actions in BrowserGym are natural language strings that describe browser operations:

```python
from envs.browsergym_env import BrowserGymAction

# Click actions
action = BrowserGymAction(action_str="click('Submit button')")
action = BrowserGymAction(action_str="click('element_id_123')")

# Type actions
action = BrowserGymAction(action_str="fill('username', 'john@example.com')")
action = BrowserGymAction(action_str="fill('password', 'secret123')")

# Navigate actions
action = BrowserGymAction(action_str="goto('https://example.com')")

# Keyboard actions
action = BrowserGymAction(action_str="press('Enter')")
action = BrowserGymAction(action_str="press('Tab')")

# Scroll actions
action = BrowserGymAction(action_str="scroll('down')")
```

### Observation

Observations contain multiple modalities:

```python
result = env.step(action)
obs = result.observation

# Text observations
print(obs.text)          # Primary text representation (AXTree or DOM)
print(obs.axtree_txt)    # Accessibility tree
print(obs.pruned_html)   # Pruned HTML (interactive elements only)

# Page metadata
print(obs.url)           # Current URL
print(obs.goal)          # Task goal/instruction

# Visual (if enabled)
if obs.screenshot is not None:
    print(obs.screenshot.shape)  # [height, width, channels]

# Error handling
if obs.last_action_error:
    print(f"Action failed: {obs.error}")

# Episode status
print(obs.done)          # True if episode ended
print(obs.reward)        # Reward for the step

# Access full BrowserGym data (includes timestamps, etc.)
print(obs.metadata["browsergym_obs"])  # Full observation dict from BrowserGym
print(obs.metadata["browsergym_info"]) # Full info dict (timestamps, page state, etc.)
```

#### Advanced: Accessing Raw BrowserGym Data

For VisualWebArena or custom training, you may need additional data like timestamps or browser state. The full BrowserGym observation and info dicts are preserved in `metadata`:

```python
result = env.step(action)

# Access timestamps (if available)
info = result.observation.metadata["browsergym_info"]
if "timestamp" in info:
    print(f"Action timestamp: {info['timestamp']}")

# Access additional observation fields
obs_dict = result.observation.metadata["browsergym_obs"]
if "dom_object" in obs_dict:
    dom = obs_dict["dom_object"]
    # Work with raw DOM object

# Access page performance data
if "performance" in info:
    print(f"Page load time: {info['performance']}")
```

### State

The environment state tracks progress:

```python
state = env.state()

print(f"Benchmark: {state.benchmark}")     # 'miniwob', 'webarena', etc.
print(f"Task: {state.task_name}")          # Task name/ID
print(f"Episode: {state.episode_id}")      # Unique episode ID
print(f"Steps: {state.step_count}")        # Number of steps taken
print(f"Total Reward: {state.cum_reward}") # Cumulative reward
print(f"Goal: {state.goal}")               # Task instruction
print(f"URL: {state.current_url}")         # Current page URL
```

## Configuration

Environment variables:

### Common Settings
- `BROWSERGYM_BENCHMARK`: Benchmark to use (`miniwob`, `webarena`, `visualwebarena`, `workarena`)
- `BROWSERGYM_TASK_NAME`: Specific task name (optional, will use first available if not set)
- `BROWSERGYM_HEADLESS`: Run browser in headless mode (default: `true`)
- `BROWSERGYM_VIEWPORT_WIDTH`: Browser viewport width (default: `1280`)
- `BROWSERGYM_VIEWPORT_HEIGHT`: Browser viewport height (default: `720`)
- `BROWSERGYM_TIMEOUT`: Action timeout in milliseconds (default: `10000`)

### WebArena-Specific (only needed for WebArena benchmark)
- `SHOPPING`: Shopping website URL
- `SHOPPING_ADMIN`: Shopping admin panel URL
- `REDDIT`: Reddit-like forum URL
- `GITLAB`: GitLab instance URL
- `MAP`: Map service URL
- `WIKIPEDIA`: Wikipedia instance URL
- `HOMEPAGE`: Homepage URL

## Supported Benchmarks

### 1. Custom Tasks (Rapid Prototyping)  For Development

- **Unlimited tasks**: Create domain-specific tasks
- **No dependencies**: No BrowserGym package needed
- **Instant iteration**: Modify HTML and logic quickly
- **Full control**: Define rewards, termination, UI
- **Fast setup**: Just add Python class and HTML file

**Use Case**: Rapid prototyping, domain-specific training, testing new ideas

**Tasks**: `copy-paste`, `copy-paste-multitab`, *[your tasks here]*

```python
env = BrowserGymEnv(environment={
    "BROWSERGYM_BENCHMARK": "custom",
    "BROWSERGYM_TASK_NAME": "copy-paste"
})
```

### 2. MiniWoB++ (Training)  Recommended for Training

- **100+ tasks** ranging from simple (click buttons) to complex (form filling, navigation)
- **Fast**: Instant resets, quick episodes
- **Randomized**: Task variations for generalization
- **No setup**: Works out-of-the-box
- **Dense rewards**: Immediate feedback for learning

**Use Case**: Train agents on fundamental web navigation skills

### 3. WebArena (Evaluation)  Benchmark

- **812 realistic tasks** across 6 websites
- **Complex**: Multi-step reasoning, real web interfaces
- **Requires setup**: Need to run 7 backend services
- **Sparse rewards**: Binary success/failure
- **Evaluation-focused**: Test real-world performance

**Use Case**: Evaluate agents on realistic web tasks

### 4. VisualWebArena (Evaluation)  Visual Benchmark

- **910 tasks** requiring visual understanding
- **Multimodal**: Both text and visual observations
- **Requires setup**: Similar to WebArena
- **Challenging**: Requires visual reasoning

**Use Case**: Test visual web navigation capabilities

### 5. WorkArena (Evaluation)  Enterprise Benchmark

- **Enterprise tasks**: CRM, project management, etc.
- **Realistic workflows**: Real enterprise software
- **Requires setup**: Enterprise software instances

**Use Case**: Evaluate on business automation tasks

## Typical Training Pipeline

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Stage 1: Train on MiniWoB (simple tasks, fast)
train_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-button",
    }
)

# Train your agent (RL, imitation learning, etc.)
agent.train(train_env, num_episodes=10000)
train_env.close()

# Stage 2: Evaluate on WebArena (complex tasks, realistic)
eval_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",
        # ... WebArena URLs
    }
)

# Test performance
success_rate = agent.evaluate(eval_env, num_tasks=812)
print(f"WebArena Success Rate: {success_rate:.2%}")
eval_env.close()
```

## Development & Testing

### Running Tests

```bash
# From the OpenEnv repository root
pytest tests/envs/test_browsergym_env.py
```

### Local Development

```bash
# Install in development mode
cd /path/to/OpenEnv
pip install -e .

# Install BrowserGym
pip install browsergym browsergym-miniwob browsergym-webarena

# Run the server locally
cd src/envs/browsergym_env/server
export BROWSERGYM_BENCHMARK=miniwob
export BROWSERGYM_TASK_NAME=click-test
python app.py
```

## Project Structure

```
browsergym_env/
 __init__.py              # Module exports
 models.py                # Action, Observation, State dataclasses
 client.py                # HTTPEnvClient implementation
 README.md                # This file
 server/
     __init__.py
     app.py               # FastAPI application
     browsergym_environment.py  # Environment implementation
     Dockerfile           # Container specification
     requirements.txt     # Python dependencies
```

## References

- [BrowserGym GitHub](https://github.com/ServiceNow/BrowserGym)
- [MiniWoB++ Paper](https://arxiv.org/abs/1802.08802)
- [WebArena Paper](https://arxiv.org/abs/2307.13854)
- [WebArena Website](https://webarena.dev/)
- [VisualWebArena Paper](https://jykoh.com/vwa)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
