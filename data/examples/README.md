# Sample policy documents

Three PDFs used to demo the analyser against documents of different
coverage profiles.

| File | Source | Profile |
| --- | --- | --- |
| `minimal_policy.pdf` | Synthetic, ~100 words (original) | Intentionally thin — most controls should come out `not_covered`. Useful for gap-detection sanity checks. |
| `sans_acceptable_use.pdf` | Synthetic, SANS-template shape (original) | Medium breadth — targets acceptable-use, password, incident-reporting, training, and compliance topics. Filename nods at SANS's template style; it is **not** a SANS template. |
| `northwestern_infosec_policy.pdf` | Real, downloaded from Northwestern University | Broad coverage — 4-page, 11-section ISO-aligned institutional information security policy. |

The two synthetic PDFs are regenerated from plain-text sources in
`sources/` by `scripts/build_samples.py`:

```
uv run python scripts/build_samples.py
```

## Attribution and licensing of sample content

- `minimal_policy.txt` and `sans_acceptable_use.txt` are original content
  written for this project. They are not reproductions of any specific
  SANS or other published template.
- `northwestern_infosec_policy.pdf` was downloaded verbatim from
  <https://policies.northwestern.edu/docs/information-security-policy.pdf>
  on 2026-04-21. Copyright is retained by Northwestern University; the
  file is included here unmodified solely as a realistic input document
  for the demo (educational / non-commercial use). If this repository is
  ever made fully public, this file should be removed in favour of a
  redistributable sample (or replaced with each reviewer downloading the
  URL themselves).

The rest of this repository (code, synthetic sample policies, sample
control corpus, documentation, tests) is under this repo's top-level
licence.
