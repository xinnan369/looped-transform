# LC-007K status

`COMPLETE / READY_FOR_REVIEW`

Issue #36 freezes one new seed-42 U4-shift-768 case on PR #34 result `c25d69b54bd141d4e0a91fdaf92e2f5580098ed1`. The owner directed launch of this exact single case. The local authorization records that PR #38 had no independent review at launch. The run launched at 10:10:56 Asia/Shanghai on 2026-11-28 from execution commit `06205f38d847f725e9a621aa276d0a0d11186d90`. The original experiment config and limits remain in force. The separate LC-007J audit has no bearing on this design.

S768 retains the accepted engineering result and the preregistered joint fail. Shared-2 aux+EMA remains the formal baseline. [Protocol](protocol.md) · [Tests](test_results.md) · [Resource plan](resource_plan.md)

The single attempt finished at 10:32:21 with all 40 epochs and 7,200 updates verified. U768 beat U512 on the two primary metrics, with FP and loss-change guards failing. S768 missed the strict paired metric directions against U768. [Training result](training_results/README.md) · [Resource audit](training_results/resource_audit.md)
