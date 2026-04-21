# Sample policy documents

Three PDFs used to demo the analyser against documents of different
coverage profiles.

| File | Source | Profile |
| --- | --- | --- |
| `minimal_policy.pdf` | Synthetic, ~100 words | Intentionally thin — most controls should come out `not_covered`. Useful for gap-detection sanity checks. |
| `sans_acceptable_use.pdf` | Synthetic, SANS-template shape | Medium breadth — targets acceptable-use, password, incident-reporting, training, and compliance topics. |
| `gitlab_infosec_excerpt.pdf` | Synthetic, handbook-style | Broad coverage — 16 sections touching access control, cryptography, supplier security, incident response, backup, vulnerability management, and more. |

All three are regenerated from plain-text sources in `sources/` by
`scripts/build_samples.py`:

```
uv run python scripts/build_samples.py
```

Editing the `.txt` source and re-running the script regenerates the PDF.
The source texts are original, non-copyrighted content loosely modelled on
the style of public policy templates — they are not copies of any specific
SANS, GitLab, or ISO document.
