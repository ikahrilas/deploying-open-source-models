from dotenv import load_dotenv

load_dotenv()


def closed_api_cost(input_tokens, output_tokens, input_price_per_m, output_price_per_m):
    """Cost of a single request to a closed API."""
    return (
        input_tokens / 1_000_000 * input_price_per_m
        + output_tokens / 1_000_000 * output_price_per_m
    )


def self_hosted_monthly(hourly_instance_cost):
    """Fixed monthly cost of running a GPU instance 24/7."""
    return hourly_instance_cost * 24 * 30


def break_even_volume(hourly_instance_cost, closed_api_cost_per_req):
    """Requests per hour where self-hosting becomes cheaper."""
    return int(hourly_instance_cost / closed_api_cost_per_req)


# GPT-5.6 Luna: typical summarization request (800 input tokens, 100 output tokens)
luna_cost = closed_api_cost(
    input_tokens=800, output_tokens=100,
    input_price_per_m=0.20, output_price_per_m=1.20
)

# Claude Haiku 4.5: same token counts
haiku_cost = closed_api_cost(
    input_tokens=800, output_tokens=100,
    input_price_per_m=1.00, output_price_per_m=5.00
)

# Self-hosted FLAN-T5 on a T4 GPU (~$0.53/hour, 24/7)
hourly_cost = 0.526
monthly_fixed = self_hosted_monthly(hourly_cost)
gpu_throughput = 600  # requests per hour on a T4

print("=== COST COMPARISON ===\n")
print(f"GPT-5.6 Luna cost per request: ${luna_cost:.6f}")
print(f"  At 10K requests/day: ${luna_cost * 10_000:.2f}/day (${luna_cost * 10_000 * 30:.0f}/month)")
print(f"\nClaude Haiku 4.5 cost per request: ${haiku_cost:.6f}")
print(f"  At 10K requests/day: ${haiku_cost * 10_000:.2f}/day (${haiku_cost * 10_000 * 30:.0f}/month)")
print(f"\nSelf-hosted FLAN-T5 (T4 GPU, 24/7): ${monthly_fixed:.0f}/month fixed")
print(f"  At {gpu_throughput} req/hour capacity: up to {gpu_throughput * 24:,} requests/day")

be = break_even_volume(hourly_cost, luna_cost)
print(f"\nBreak-even vs GPT-5.6 Luna: ~{be:,} requests/hour (~{be * 24:,}/day)")
print(f"  Below this, Luna is cheaper. Above this, self-hosting is cheaper.")