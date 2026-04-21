# ISO 27001:2022 Annex A — control description prompt

This file documents how the paraphrased control descriptions in
`iso27001_annex_a.yaml` were produced, for reproducibility and honesty.

## Sources

Descriptions are paraphrased from publicly available implementation guidance
(ENISA materials, Big 4 implementation guides, vendor control libraries from
Vanta / Drata / Secureframe, ISMS.online). They are **not** the normative
text of ISO 27001:2022 Annex A or ISO 27002:2022 — both are copyrighted by
ISO and redistributed under licence only. Control identifiers and short titles
are factual references and are not copyrightable.

## Prompt used for the initial draft

The YAML was drafted by prompting a local LLM with the following template,
one batch per theme, and then hand-reviewed against the sources above.

````
Generate YAML entries for the following ISO 27001:2022 Annex A controls.

Required fields per entry:
  - id:          the control identifier, e.g. "A.5.15"
  - title:       the official short title (factual reference)
  - theme:       one of Organizational | People | Physical | Technological
  - description: 2–3 sentences paraphrased from public implementation
                 guidance. Describe the control's purpose and what a well
                 implemented control looks like in practice. Do NOT copy
                 ISO's normative text. Keep the description
                 technology-neutral. Do not invent specific tools or vendor
                 names.
  - queries:     an empty list ([]) — will be populated later by
                 the multi-query generation step.

Controls to cover: {control_ids}

Output valid YAML only, no commentary. Preserve the order of controls.
````

## Post-generation review

Each entry was spot-checked against at least one public source to ensure:

- Control ID and title match ISO 27001:2022 Annex A (93 controls total,
  organised into four themes — Organizational / People / Physical /
  Technological).
- The description is technology-neutral and does not name specific
  products or vendors.
- No hallucinated controls (e.g. invented IDs) slipped in.
- The `theme` matches the control's assigned theme in the standard.

The curated subset is ~30 controls across the four themes. Expanding to the
full 93 is a matter of appending additional entries — the pipeline treats
the YAML as the single source of truth.
