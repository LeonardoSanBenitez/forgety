"""I-CARE MCP server.

Exposes I-CARE Result Templates as MCP tools so LLMs can query benchmark results
natively. Runs as a sidecar container (port 80 inside Docker, mapped to 8002 on host).

Transport: Streamable HTTP at /mcp
Claude Desktop config (via mcp-remote):
    {
      "mcpServers": {
        "icare": {
          "command": "npx",
          "args": ["-y", "mcp-remote", "http://localhost:8002/mcp"]
        }
      }
    }
"""
import libs.env  # normalizes HF_TOKEN before any import below can read it; see libs/env.py

import inspect
import io
import json
import logging
from typing import Any, Dict

import uvicorn
from mcp.server.fastmcp import FastMCP, Image

from vision_unlearning.benchmarks.I_care import (
    rt_name_to_class,
    rt_name_to_params,
    convert_params_from_gui_to_backend,
)
from vision_unlearning.datasets.testbed import get_metadata_filtered

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I-CARE Benchmark Glossary (injected into every _interpretation)
# ---------------------------------------------------------------------------
_ICARE_GLOSSARY = """\
## I-CARE Benchmark — Key Concepts

- **Entity**: A distinct semantic concept being studied (e.g., "George W. Bush", "schnauzer").
- **Task**: A group of related entities. Valid values: "people", "scenes", "breeds", "objects".
- **Unlearning algorithm**: Technique that removes entity knowledge from a model.
  Valid values: "distil" (FADE/distil), "uce" (UCE), "munba" (MUNBA).
- **Mp (pair-level metric)**: Measured for an (emitter, receiver) pair — how much unlearning
  the emitter degrades the receiver's image quality.
  - clip_diff: CLIP score difference (on − off). More negative = worse = more interference.
  - dino_diff: DINOv2 similarity difference (on − off). More negative = worse.
  - ssim: Structural similarity difference (on − off). More negative = worse.
- **Me (entity-level metric)**: Mp aggregated per emitter across all receiver pairs.
  - clip_diff: Mean CLIP degradation caused by this entity's unlearning.
  - retain_average_clip_diff: Average degradation of retained entities.
  - embedding_specificity_ratio: How targeted unlearning is in embedding space (>1 = good).
- **Emitter**: The entity being unlearned (causes interference in retained entities).
- **Receiver**: An entity whose quality degrades as collateral damage.
- **Interference**: Unintended degradation of retained concepts after unlearning.
"""

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "I-CARE Benchmark",
    instructions=(
        "This server provides access to the I-CARE machine unlearning benchmark. "
        "I-CARE evaluates how well unlearning methods surgically remove specific entities "
        "from text-to-image models without degrading unrelated retained entities (interference). "
        "Start by calling list_rts() to discover available analyses, or list_entities(task) "
        "to see what entities exist in a task."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _example_value(param: str) -> Any:
    """Return an illustrative example value for a well-known parameter name."""
    _EXAMPLES: Dict[str, Any] = {
        "model": "sd1.4",
        "task": "people",
        "unlearning_algorithm": "distil",
        "interference_entity": "clip_diff",
        "interference_entity_1": "clip_diff",
        "interference_entity_2": "retain_average_clip_diff",
        "interference_pair": "clip_diff",
        "similarity_metric": "jacc",
        "attribute": "gender",
        "attribute_value": "female",
        "entity": "George W. Bush",
        "attribute_1": "gender",
        "attribute_2": "age_group",
        "latent_embedding": "clip",
        "entity_1": "George W. Bush",
        "entity_2": "Tony Blair",
        "interference_entity_list": ["clip_diff", "retain_average_clip_diff"],
        "attribute_list": ["gender", "age_group"],
        "unlearning_algorithm_list": ["distil", "uce"],
    }
    return _EXAMPLES.get(param, f"<{param}>")


def _build_interpretation(rt_name: str, data: Dict[str, Any]) -> str:
    """Build _interpretation string from RT docstring + key result numbers."""
    cls = rt_name_to_class[rt_name]
    docstring = inspect.getdoc(cls) or ""

    # Extract the **Interpretation** section if present in the docstring.
    interp_section = ""
    if "**Interpretation**" in docstring:
        start = docstring.index("**Interpretation**")
        chunk = docstring[start:]
        next_header = chunk.find("\n**", 5)
        interp_section = (
            chunk[:next_header].strip() if next_header > 0 else chunk.strip()
        )

    # Collect key numeric fields from the result dict.
    result = data.get("result", {})
    summary_items = []
    if isinstance(result, dict):
        for key in [
            "n_total", "n_valid", "mean_ratio", "std_ratio",
            "fraction_above_1", "pearson_r", "pearson_pvalue",
            "spearman_r", "spearman_pvalue",
        ]:
            val = result.get(key)
            if val is not None:
                fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                summary_items.append(f"{key}={fmt}")

    parts = [
        f"## Result Template: {rt_name}",
        _ICARE_GLOSSARY,
    ]
    if interp_section:
        parts.append(f"## RT-Specific Interpretation\n{interp_section}")
    if summary_items:
        parts.append("## Key Result Numbers\n" + ", ".join(summary_items))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_rts() -> str:
    """List all available I-CARE Result Templates with descriptions and required parameters.

    Call this first to discover what analyses are available and what parameters each
    RT requires before calling compute_rt() or get_rt_figure().

    Returns a JSON catalogue: {rt_name: {description, required_params, example}}.
    """
    catalogue: Dict[str, Any] = {}
    for name, cls in rt_name_to_class.items():
        params = rt_name_to_params.get(name, [])
        docstring = inspect.getdoc(cls) or "No description available."
        # First paragraph of docstring only (brief description).
        short_doc = docstring.split("\n\n")[0].strip()
        catalogue[name] = {
            "description": short_doc,
            "required_params": params,
            "example_params": {p: _example_value(p) for p in params},
        }
    return json.dumps(catalogue, indent=2)


@mcp.tool()
def list_entities(task: str) -> str:
    """List entity names and attribute keys for a given task.

    Use this to discover which entities exist in a task before calling RTs that
    require an entity name (e.g., EmbeddingUnlearningProfile, InterferenceVisualSummary).

    Args:
        task: One of "people", "scenes", "breeds", "objects".

    Returns JSON with entity_names (list of strings) and attribute_keys (list of
    attribute names that can be used in RTs like SignificantRelationshipCategorical).
    """
    try:
        metadata = get_metadata_filtered(task)  # type: ignore[arg-type]
    except FileNotFoundError:
        return json.dumps({
            "error": (
                f"Metadata file for task '{task}' not found. "
                "Valid tasks: people, scenes, breeds, objects."
            )
        })
    except Exception as exc:
        logger.exception("Error loading metadata for task %s", task)
        return json.dumps({"error": f"Could not load metadata: {exc}"})

    entity_names = [m["name"] for m in metadata]
    attribute_keys = sorted({k for m in metadata for k in m.keys() if k != "name"})

    return json.dumps({
        "task": task,
        "n_entities": len(entity_names),
        "entity_names": entity_names,
        "attribute_keys": attribute_keys,
    }, indent=2)


@mcp.tool()
def compute_rt(rt_name: str, params: str) -> str:
    """Compute an I-CARE Result Template and return numeric results as JSON.

    Returns data and a natural-language interpretation. Does NOT include figures.
    Call get_rt_figure() if you want the visual output.

    Args:
        rt_name: Name of the RT. Call list_rts() to see all available RTs and
                 their required parameters.
        params: JSON string of parameters, e.g.
                '{"task": "people", "unlearning_algorithm": "distil"}'
                The "model" parameter defaults to "sd1.4" when omitted.

    Returns JSON with:
        - result: numeric RT outputs (metrics, correlations, entity lists, etc.)
        - metadata: the RT parameters that produced this result
        - _interpretation: natural-language explanation including I-CARE glossary
    """
    if rt_name not in rt_name_to_class:
        available = sorted(rt_name_to_class.keys())
        return json.dumps({
            "error": f"Unknown RT '{rt_name}'.",
            "available_rts": available,
        })

    try:
        params_dict: Dict[str, Any] = json.loads(params)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON in params: {exc}"})

    # Apply GUI→backend mapping for safety (passes through canonical values unchanged).
    params_dict = convert_params_from_gui_to_backend(params_dict)

    cls = rt_name_to_class[rt_name]
    try:
        rt = cls(**params_dict)
        data = rt.compute()
    except Exception as exc:
        logger.exception("Error computing RT %s with params %s", rt_name, params_dict)
        return json.dumps({"error": f"Compute error: {exc}"})

    data["_interpretation"] = _build_interpretation(rt_name, data)
    return json.dumps(data, default=str)


@mcp.tool()
def get_rt_figure(rt_name: str, params: str) -> Image:
    """Compute an I-CARE Result Template and return the result figure as an image.

    The figure will be rendered inline in your conversation. Use compute_rt() first
    to see the numeric results, then call this for the visual summary.

    Args:
        rt_name: Name of the RT. Call list_rts() to see all available RTs.
        params: JSON string of parameters, e.g.
                '{"task": "people", "unlearning_algorithm": "distil"}'

    Returns a PNG image.
    """
    if rt_name not in rt_name_to_class:
        raise ValueError(
            f"Unknown RT '{rt_name}'. "
            f"Available: {sorted(rt_name_to_class.keys())}"
        )

    try:
        params_dict: Dict[str, Any] = json.loads(params)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in params: {exc}") from exc

    params_dict = convert_params_from_gui_to_backend(params_dict)

    cls = rt_name_to_class[rt_name]
    rt = cls(**params_dict)
    data = rt.compute()

    fig_result = cls.plot(data, return_fig=True)
    if fig_result is None:
        raise ValueError(
            f"RT '{rt_name}' plot() returned None. "
            "This RT may not support figure output."
        )
    fig, _ = fig_result

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return Image(data=buf.read(), format="png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        # Claude Desktop mode: communicate via stdin/stdout inside the container.
        # Claude Desktop config (container must already be running):
        #   {
        #     "mcpServers": {
        #       "icare": {
        #         "command": "docker",
        #         "args": ["exec", "-i", "platform-mcp", "python", "app/server.py", "--stdio"]
        #       }
        #     }
        #   }
        mcp.run(transport="stdio")
    else:
        # Default: HTTP server mode (used inside Docker container on port 80).
        uvicorn.run(
            mcp.streamable_http_app(),
            host="0.0.0.0",
            port=80,
            log_level="info",
        )
