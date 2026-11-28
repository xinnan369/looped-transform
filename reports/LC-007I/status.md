# LC-007I status

`COMPLETE / READY_FOR_REVIEW`

Issue #33 narrows the experiment to one seed-42 Shifted-Shared-4 run. The CPU freeze verifies cached data, separate training and evaluation geometry, archived-reference parity and the launch and archival paths. The owner directed immediate launch and waived independent prelaunch review for this attempt. The single attempt launched on 2026-11-28 at 08:08:07 Asia/Shanghai using execution commit `8dd00d0b9f78cc954a92824f868825ed786f4fd1`. The execution config hash, training recipe and limits match the original freeze.

The queue completed at 08:27:46 with all 40 epochs and 7,200 updates verified. MAE and nAP improved against the historical 512 crop. The false-positive and loss stability checks failed, giving a joint fail. [Training results](training_results/README.md) · [Resource audit](training_results/resource_audit.md)

The branch stacks on PR #32 result `c8027284084e982b3e7b468a919769e9aeb0e5bc`. That run passed retrospective technical acceptance and failed all four research gates. The independent prelaunch review is absent from its PR history. LC-007I records the implementer recheck and the owner's explicit waiver as separate facts. CGCR is paused and Shared-2 aux+EMA remains the formal baseline. [Parent technical acceptance](https://github.com/xinnan369/looped-transform/pull/32#pullrequestreview-6084534441)

[Protocol](protocol.md) · [CPU tests](test_results.md) · [Resource plan](resource_plan.md)
