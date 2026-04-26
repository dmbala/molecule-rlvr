# Episode schema - `data/prompts/*_episodes.jsonl`

Each line is one multi-turn task instance for a tool-using molecular design
agent. Unlike `schema.md`, which describes single prompts, this schema treats
one record as an entire episode template with turn state carried forward by the
runner.

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable unique id such as `bace1_leadopt_0001` |
| `target` | string | Target key from `verifier/target_registry.py` such as `bace1`, `mpro_noncovalent`, or `plpro` |
| `task` | string | One of `de_novo`, `lead_opt`, `repair` |
| `split` | string | `train`, `eval`, or `test` |
| `instruction` | string | Natural-language task given to the model at turn 1 |
| `context` | string | Target-pocket or assay context shown to the model |
| `seed_smiles` | string \| null | Starting scaffold for `lead_opt`, or the prior failed molecule for `repair` |
| `seed_provenance` | string | Source tag such as `none`, `known_ligand_fragment`, `zinc_random`, `prior_failed_candidate` |
| `constraints` | object | Structured constraints and thresholds; see below |
| `success` | object | Terminal success rule used by offline eval and RL logging |
| `max_turns` | int | Maximum number of model turns in the episode, usually 2-5 |
| `tool_budget` | int | Optional cap on verifier/tool invocations inside the episode |
| `metadata` | object | Free-form bookkeeping: scaffold family, literature source, assay notes |

## `constraints`

Recommended fields:

| Field | Type | Description |
|-------|------|-------------|
| `dock_max` | float \| null | Success threshold such as `-8.0`; `null` if docking is not used |
| `qed_min` | float \| null | Minimum acceptable QED |
| `sa_max` | float \| null | Maximum acceptable SA score |
| `mw_max` | float \| null | Maximum molecular weight |
| `logp_max` | float \| null | Optional upper bound on logP |
| `hbd_max` | int \| null | Optional donor cap |
| `hba_max` | int \| null | Optional acceptor cap |
| `novelty_max_tanimoto_to_seed` | float \| null | Prevent trivial copies of the seed |
| `non_covalent_only` | bool | If true, covalent warheads or reactive groups should fail |
| `forbid_pains` | bool | If true, PAINS alerts are disallowed |

## `success`

`success` is the explicit terminal condition for reporting and early-stop reward:

| Field | Type | Description |
|-------|------|-------------|
| `require_valid` | bool | Molecule must parse successfully |
| `require_all_constraints` | bool | All non-null constraint fields must pass |
| `min_improvement_vs_seed` | float \| null | Optional minimum improvement in docking score over the seed |
| `max_tool_calls` | int \| null | Optional success cap on oracle usage |

## Turn feedback record

The episode runner should store one feedback record per completed turn and feed
the latest one back into the model prompt.

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Parent episode id |
| `turn` | int | 1-indexed completed turn |
| `candidate_id` | string | Stable id like `bace1_leadopt_0001_t2` |
| `parent_id` | string \| null | Previous candidate id for `edit` actions |
| `action` | string | Parsed model action, usually `propose` or `edit` |
| `smiles` | string \| null | Final parsed candidate for the turn |
| `summary_text` | string | Short natural-language feedback surfaced to the model |
| `tool_feedback` | object | Structured verifier outputs for the turn |
| `done` | bool | Whether the episode terminated after this turn |
| `success` | bool | Whether success criteria were met on this turn |

Recommended `tool_feedback` fields:
- `valid`
- `failure_reason`
- `dock_kcal`
- `qed`
- `sa_raw`
- `mw`
- `logp`
- `hbd`
- `hba`
- `novelty_to_seed`
- `similarity_to_parent`
- `passes_all_constraints`
- `improved_vs_parent`
- `remaining_turns`

## Allowed model output tags

To keep reward parsing deterministic, the model should emit:

```text
<action>propose</action>
<answer>CCOc1ncc(...)</answer>
```

For edit trajectories:

```text
<action>edit</action>
<parent_id>bace1_leadopt_0001_t1</parent_id>
<answer>CCOc1ncc(...)</answer>
```

Optional reasoning is allowed but should not be required by the parser:

```text
<reasoning>...</reasoning>
<action>edit</action>
<parent_id>bace1_leadopt_0001_t1</parent_id>
<answer>CCOc1ncc(...)</answer>
```

## Example episode record

```json
{
  "id": "bace1_leadopt_0001",
  "target": "bace1",
  "task": "lead_opt",
  "split": "train",
  "instruction": "Starting from the seed scaffold, improve BACE1 docking while keeping QED > 0.5 and SA < 5. Use tool feedback to revise over multiple turns.",
  "context": "BACE1 catalytic site. Favor productive H-bonding near Asp32/Asp228 and avoid oversized lipophilic substitutions.",
  "seed_smiles": "CCOc1ncc(NC(=O)Nc2ccccc2)s1",
  "seed_provenance": "known_ligand_fragment",
  "constraints": {
    "dock_max": -8.0,
    "qed_min": 0.5,
    "sa_max": 5.0,
    "mw_max": 550.0,
    "logp_max": 5.5,
    "novelty_max_tanimoto_to_seed": 0.85,
    "non_covalent_only": true,
    "forbid_pains": true
  },
  "success": {
    "require_valid": true,
    "require_all_constraints": true,
    "min_improvement_vs_seed": 0.5,
    "max_tool_calls": 4
  },
  "max_turns": 4,
  "tool_budget": 4,
  "metadata": {
    "target_family": "aspartyl_protease",
    "task_source": "lead_opt_seeded"
  }
}
```

## Design notes

- `de_novo` episodes set `seed_smiles = null`.
- `lead_opt` episodes should dominate the train split if the project emphasizes
  iterative optimization over pure exploration.
- `repair` episodes should include an intentionally weak or constraint-violating
  starting molecule so the model must learn to recover from feedback.
- Eval/test episodes should be frozen before RL and should not share exact seeds
  with the train split.
