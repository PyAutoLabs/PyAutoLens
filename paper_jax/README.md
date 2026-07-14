# PyAutoLens-JAX JOSS paper

This directory contains the second PyAutoLens paper for submission to the
[Journal of Open Source Software](https://joss.theoj.org/). It is intentionally
separate from `../paper/`, which contains the published 2021 PyAutoLens paper.

## Files

- `paper.md` — manuscript and JOSS metadata.
- `paper.bib` — bibliography cited by the manuscript.
- `paper.pdf` — local build output; do not commit it.

## Drafting checklist

- Confirm the full author list, affiliations, ORCIDs, corresponding author, and
  submission date.
- Expand the manuscript to the current JOSS target of 750–1750 words.
- Replace every drafting comment with specific, evidenced prose.
- Compare against relevant strong- and weak-lensing software in “State of the
  field”.
- Include reproducible GPU benchmarks and distinguish compilation from
  steady-state execution.
- Add concrete evidence of research impact and verify every bibliography entry.
- Keep the AI usage disclosure accurate as the manuscript evolves.

The current format requirements are documented in the
[JOSS paper guide](https://joss.readthedocs.io/en/latest/paper.html).

## Build the paper

From the PyAutoLens repository root, compile with the official JOSS Inara image:

```bash
docker run --rm \
  --volume "$PWD/paper_jax:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

The generated PDF is written to `paper_jax/paper.pdf`.
