# ACU economics (illustrative)

This is a demo model, not Cognition's actual ACU formula or pricing. Every
parameter can be set through an environment variable.

| Parameter | Default | Env var | Basis |
|---|---|---|---|
| Customer price per ACU | $2.25 | `ACU_PRICE_USD` | devin.ai/pricing (Core tier, pay-as-you-go) |
| AWS COGS per ACU | $0.90 | `AWS_COGS_PER_ACU_USD` | assumption: ~40% COGS |
| AWS revenue share | 20% | `AWS_REVENUE_SHARE` | assumption: negotiated first-party / OEM share |
| Agent runtime | $0.40/hr | `RUNTIME_USD_PER_HOUR` | ~m7i.2xlarge on-demand |
| Other AWS services | 10% of infra | `OTHER_AWS_PCT_OF_INFRA` | S3, networking, logging |
| Bedrock token prices | see `config.py` | `BEDROCK_PRICING_JSON` | Bedrock on-demand list prices; verify per region |

```
bedrock_usd  = Σ tokens_in × price_in + tokens_out × price_out
infra_usd    = (bedrock_usd + runtime_usd) × (1 + other_pct)
acus         = infra_usd / aws_cogs_per_acu
revenue      = acus × acu_price                 # customer → AWS bill
aws_share    = revenue × aws_revenue_share      # AWS keeps
aws_capture  = aws_share + infra_usd            # AWS total
cognition    = revenue − aws_share − infra_usd
```

Worked example with the defaults: 1 ACU → $2.25 customer revenue → $0.45 AWS
share + $0.90 Cognition infra bill to AWS = **$1.35 AWS capture per ACU (60%)**;
$0.90 Cognition contribution. At 10M ACUs/yr that is $13.5M/yr AWS capture:
$4.5M revenue share plus a $9.0M infra bill. How much of the $9.0M is Bedrock
depends on model mix and runtime share; the dashboard shows the measured split for each run.
