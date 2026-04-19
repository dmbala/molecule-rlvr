# Prompt schema — data/prompts/*.jsonl

Each line is one prompt with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable unique id (`mpro_<zero-padded-index>`) |
| `split` | string | `train` or `eval` — eval split is never seen during RL rollouts |
| `context` | string | Pocket description / structural context. Fixed small vocabulary of residues. |
| `seed_smiles` | string \| null | Starting molecule (lead-optimization mode per Fix 8.2), or null for de novo |
| `seed_provenance` | string | One of `none`, `known_inhibitor_fragment`, `zinc_random`, `prior_turn` |
| `instruction` | string | Task wording given to the model |
| `reward_strictness` | string | `loose` (<-6), `medium` (<-7), `tight` (<-8) — surfaces to the model so it can calibrate |
| `pocket_focus` | string | `all`, `catalytic_dyad`, `s1`, `s4` — varies the residue emphasis |
| `instruction_style` | string | `terse`, `detailed`, `constraints_first` |
| `turn` | int | 1-indexed turn number within a multi-turn trajectory |

## Variation axes (Fix 6)

Prompts are generated as the Cartesian product of:
- **pocket_focus** ∈ {all, catalytic_dyad, s1, s4}
- **seed_provenance** ∈ {none, known_inhibitor_fragment, zinc_random, prior_turn}
- **instruction_style** ∈ {terse, detailed, constraints_first}
- **reward_strictness** ∈ {loose, medium, tight}

Full product = 4 × 4 × 3 × 3 = 144 distinct templates; instantiate each with
multiple concrete seeds to reach 300 total. 240 train / 60 eval split.

## Fix 8.2 compliance
- ≥ 80 % of prompts must have `seed_smiles != null` (lead-optimization bias).
- ~20 % retain `seed_provenance = none` to preserve exploration breadth.

## Example

```json
{"id":"mpro_0001","split":"train","context":"SARS-CoV-2 Mpro active site. Catalytic dyad: His41, Cys145. S1 pocket residues: His163, Glu166, Phe140.","seed_smiles":"O=C(NC1CCOCC1)c1ccc(F)cc1","seed_provenance":"known_inhibitor_fragment","instruction":"Starting from the scaffold above, propose a modified SMILES that improves Vina docking score against SARS-CoV-2 Mpro while keeping QED > 0.5 and SA < 5. Briefly justify your edit using pocket residues.","reward_strictness":"medium","pocket_focus":"catalytic_dyad","instruction_style":"detailed","turn":1}
```
