# LC-007H status

`E40_DIAGNOSTIC_COMPLETE / AWAITING_ACCEPTANCE`

CPU preparation completed against reviewed PR #30 head `4a0f7b023e314af8ac6fe3c8ea0390d3428dee63`. The dedicated E40 driver freeze is `f52546b39a15f6419568691c2777d2645392b1be`. After launch review and separate approval, execution at `856dfb30eec4987849f8809575382f59b750c21f` completed all four checkpoint sweeps. The parity checks passed for all 160 stage-image rows and 16 grouped rows per checkpoint. Charged time was 39.77 s and peak reserved VRAM was 224 MiB, with zero training updates and zero downloads. [Resource audit](resource_audit.md)

On the four frozen controls, the highest ambiguity bin had 21.3% later recovery among midpoint misses, versus 4.5% in the lowest bin. More than half of midpoint misses were in the two lower bins, and all owned midpoint candidates above .30 were in the highest bin. The result supports review of the recovery hypothesis before any training decision. [Uncertainty diagnostic](uncertainty_diagnostic.md) · [Research decision](research_decision.json)

Review the [archive](archived_error_diagnostic.md), [CPU tests](test_results.md), [protocol](protocol.md) and [read-only diagnostic proposal](resource_plan.md).
