# Product Guidelines

## Documentation & Tone
- **Style:** Academic and Formal.
- **Focus:** The documentation should maintain a high standard of rigor suitable for accompanying research publications, while clearly explaining technical implementation details.

## Contribution & Governance
- **Philosophy:** Community-Driven.
- **Process:** We encourage open discussion via GitHub Issues to propose and refine new features or model implementations before coding begins.
- **Standards:** While community-driven, contributions must adhere to the project's architectural patterns and pass standard quality checks.

## Visual Representation & Interpretability
- **Publication-Ready Assets:** All static plots (training curves, confusion matrices) must be high-quality and suitable for publication (using libraries like Seaborn/Matplotlib).
- **Interactive Monitoring:** Real-time training metrics should be trackable via interactive dashboards (e.g., Weights & Biases).
- **Interpretability:** Standardized heatmap generation is required for model explainability.
- **Interoperability:** Heatmaps must be exportable as GeoJSON files to support overlay on pathology images within QuPath, facilitating clinical review and validation.
