# Stripe World

Stripe World is a stateful copy of Stripe's Billing and Payments API that runs as an OpenEnv
environment. Agents act through the same tools as Stripe's MCP server. Every connection gets its
own private Stripe account, which can start from a saved state, and every database row the episode
changed is returned by `state()` for grading.

The environment is maintained outside this repository:

- Hugging Face Space: [scosman/stripe_world](https://huggingface.co/spaces/scosman/stripe_world)
- Source: [Kiln-AI/stripe_world](https://github.com/Kiln-AI/stripe_world)

It is built with [Seahaven](https://github.com/Kiln-AI/Seahaven). Stripe World is not affiliated
with Stripe.

## What it models

- **API surface:** 155 Stripe API operations across 24 tables, pinned to Stripe API version
  `2026-08-26.dahlia`. Object shapes follow Stripe's OpenAPI spec.
- **Resources:** customers, payment methods, products, prices, coupons, promotion codes, tax rates,
  payment intents, charges, refunds, disputes, setup intents, balance and balance transactions,
  payouts, subscriptions, subscription items and schedules, invoices, invoice items, credit notes,
  customer balance transactions and events.
- **Billing behaviour:** subscription status transitions, trials and cancellation; invoices that
  move from draft to open to paid, void or uncollectible; proration; dunning; credit notes; and a
  balance ledger. Stripe's test cards and tokens (`tok_visa`,
  `tok_visa_chargeDeclinedInsufficientFunds`, ...) succeed, decline and dispute as they do in
  Stripe's test mode.
- **Agent tools:** 8 of the 10 tools on Stripe's MCP server: `list_available_accounts_or_orgs`,
  `get_stripe_account_info`, `manage_stripe_accounts`, `stripe_analytics`, `stripe_api_read`,
  `stripe_api_write`, `stripe_api_search` and `stripe_api_details`.

Webhook delivery and the parts of Stripe outside Billing and Payments are not modelled.

## Episodes and grading

- `reset()` creates a fresh account. Pass `fixture` to start from a saved state (the repository
  ships `empty`), `seed` to make ids and timestamps reproducible, and
  `startup={"livemode": False}` for a test-mode account.
- `step()` takes `list_tools` and `call_tool` actions.
- The environment does not compute a reward. `state()` returns the change log of the episode
  under `["state"]["db"]["log"]`: one entry per changed row, with its table, operation, and the
  row before and after. Grade the episode from those rows.

## Quick start

Connect to the Space with OpenEnv's generic client. Only the `openenv` package is needed on the
client side:

```python
from openenv import AutoEnv

with AutoEnv.from_hub("scosman/stripe_world", skip_install=True) as env:
    env.reset(seed=7, startup={"livemode": False})
    tools = env.step({"type": "list_tools"}).observation["tools"]
    account = env.step(
        {"type": "call_tool", "tool_name": "list_available_accounts_or_orgs", "arguments": {}}
    ).observation["result"]["accounts"][0]
    created = env.step(
        {
            "type": "call_tool",
            "tool_name": "stripe_api_write",
            "arguments": {
                "stripe_api_operation_id": "PostCustomers",
                "parameters": {"email": "jenny@example.com"},
                "stripe_context": account["stripe_context"],
                "livemode": account["livemode"],
            },
        }
    ).observation
    print(created["result"], env.state())
```

Pass `skip_install=True`. Without it, `from_hub` tries to install the Space's repository as a
Python package, which fails for this environment. For a server you run yourself, add
`base_url="http://localhost:8000"`.

## Running the server locally

```bash
git clone https://github.com/Kiln-AI/stripe_world.git
cd stripe_world
uv sync
uv run --extra serve seahaven serve
```

The server listens on port 8000. Open `http://127.0.0.1:8000/console` to drive an instance by
hand. The [Stripe World README](https://github.com/Kiln-AI/stripe_world#readme) has
an async example with the typed `SeahavenClient` that runs and grades 100 episodes in parallel,
and shows how to save your own fixtures.
