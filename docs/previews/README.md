# Offline Regression Previews

All worlds here are hand-authored fixtures, rendered through the same validation
and pygame pipeline used by the LLM providers. No live API was called to create
these previews. See `manifest.json` for object counts and provenance.

| Scenario | World State | Map |
| --- | --- | --- |
| Forest village | [JSON](dark_forest_village.json) | [PNG](dark_forest_village.png) |
| Snow castle | [JSON](snow_castle.json) | [PNG](snow_castle.png) |
| Desert ruins | [JSON](desert_ruins.json) | [PNG](desert_ruins.png) |
| Apocalyptic city | [JSON](apocalyptic_city.json) | [PNG](apocalyptic_city.png) |
| Magic academy | [JSON](magic_academy.json) | [PNG](magic_academy.png) |

Regenerate from the repository root:

```bash
gameworldlm examples --provider fixture --output-dir docs/previews
```
