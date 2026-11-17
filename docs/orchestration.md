# Agent orchestration and background execution

Read the workspace `AGENTS.md` for role/permission rules and [AGY/Colab workflow](agy_colab_training_workflow.md) for current operating detail. The old fixed ViT-design→export topology did not describe the implemented recognition experiments.

- User defines source, budget, dataset constraints and authorization.
- Sol6/Codex plans, dispatches and reviews evidence.
- Existing AGY session performs authorized heavy rendering/training/evaluation; do not spawn extra agents unless requested or required by applicable instructions.
- A hidden local controller manages detached Colab execution, incremental downloads, hashes, recovery and callbacks. No repeatedLLMpolling.

Discover agents with `herdr agent list` / `herdr pane list`; verify identity before dispatch. Pane IDs/modelnames from old reports are examples, not defaults. Sendtasks/callbacks with `herdr agent prompt`; callbacks omit `--wait`. LatestColabskill resolves liveAGYpanes dynamically for repair notifications; coordinator notifications require explicitrouting in the task. Never use arbitrary terminalpaste asagentcommunication or restartHerdr/alterACLs to bypass permissions.

Taskacceptance is not launch. A launchACK requires actualhardware, realdata/e2esmoke and firstoptimizerupdate. Evaluationfailure afterretrieval doesnotmeantrainingfailed; repairlocal evaluation withoutretraining. Preserve source/checkpoints/labels, singlejobownership and fullresume state. Noautomaticpromotion/cleanup/Gitpush beyondauthorization.
