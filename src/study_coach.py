from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from llm_client import LLMClient
from resource_retrieval import Resource, build_queries_from_gaps, search_arxiv


@dataclass(frozen=True)
class CoachInputs:
    goal: str
    hours_per_week: int
    weeks: int
    math_score: int
    reading_score: int
    writing_score: int
    preferred_style: str


def _gap_labels(math_score: int, reading_score: int, writing_score: int) -> list[str]:
    gaps: list[str] = []
    if math_score < 70:
        gaps.append("math")
    if reading_score < 70:
        gaps.append("reading")
    if writing_score < 70:
        gaps.append("writing")
    if not gaps:
        # still provide resources for stretch goals
        gaps = ["math", "reading", "writing"]
    return gaps


def _risk_level(avg: float) -> str:
    if avg < 50:
        return "At-risk"
    if avg < 80:
        return "Average"
    return "High-performing"


def diagnose(inputs: CoachInputs) -> dict[str, Any]:
    avg = (inputs.math_score + inputs.reading_score + inputs.writing_score) / 3.0
    gaps = _gap_labels(inputs.math_score, inputs.reading_score, inputs.writing_score)

    strengths: list[str] = []
    if inputs.math_score >= 80:
        strengths.append("math")
    if inputs.reading_score >= 80:
        strengths.append("reading")
    if inputs.writing_score >= 80:
        strengths.append("writing")

    return {
        "average_score": round(avg, 2),
        "risk_level": _risk_level(avg),
        "gaps": gaps,
        "strengths": strengths,
    }


def retrieve_resources(gaps: list[str], *, per_gap: int = 2) -> list[Resource]:
    resources: list[Resource] = []
    for q in build_queries_from_gaps(gaps):
        try:
            resources.extend(search_arxiv(q, max_results=per_gap))
        except Exception:
            # retrieval is best-effort; UI will still work
            continue
    # de-dup by url
    seen: set[str] = set()
    out: list[Resource] = []
    for r in resources:
        if r.url and r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out[: max(3, per_gap * 2)]


def _prompt(inputs: CoachInputs, diag: dict[str, Any], resources: list[Resource]) -> str:
    res_lines = "\n".join(
        f"- {r.title} ({r.url})\n  Summary: {r.summary[:300]}..."
        for r in resources[:6]
    )
    return f"""
You are an AI Study Coach for a student.

Write a structured report with EXACT section headings:
Learning diagnosis
Personalized study plan
Weekly goals
Recommended learning resources (URLs)
Progress feedback / next steps

Constraints:
- Be specific and actionable.
- Use the student's goal and available study time.
- Keep the plan realistic and split by week.
- Mention how to measure progress each week.
- Use the provided resources as URLs (do not invent links).

Student goal: {inputs.goal}
Preferred study style: {inputs.preferred_style}
Time available: {inputs.hours_per_week} hours/week
Time horizon: {inputs.weeks} weeks

Current performance:
- Math score: {inputs.math_score}/100
- Reading score: {inputs.reading_score}/100
- Writing score: {inputs.writing_score}/100
- Average: {diag["average_score"]}/100
- Risk level: {diag["risk_level"]}
- Main gaps: {", ".join(diag["gaps"])}
- Strengths: {", ".join(diag["strengths"]) if diag["strengths"] else "none identified"}

Resources you may cite:
{res_lines if res_lines else "- (none retrieved)"}
""".strip()


def generate_coach_report(inputs: CoachInputs) -> dict[str, Any]:
    diag = diagnose(inputs)
    resources = retrieve_resources(diag["gaps"])

    # LLM generation is best-effort; fall back to template if missing.
    llm_text = None
    llm_provider = None
    try:
        llm = LLMClient()
        result = llm.generate(_prompt(inputs, diag, resources))
        llm_text = result.text
        llm_provider = result.provider
    except Exception:
        llm_text = None
        llm_provider = None

    if not llm_text:
        # Safe fallback that still satisfies the required structure.
        weekly = []
        focus = diag["gaps"][:]
        for w in range(1, inputs.weeks + 1):
            f = focus[(w - 1) % len(focus)]
            weekly.append(
                f"- Week {w}: Focus on {f}. Do 3 sessions (~{max(1, inputs.hours_per_week // 3)}h each): "
                f"(1) concept review, (2) guided practice, (3) timed practice + error log."
            )
        res_md = "\n".join(f"- {r.title}: {r.url}" for r in resources) if resources else "- No resources retrieved. Try again or broaden keywords."

        llm_text = f"""Learning diagnosis
- Average score: {diag['average_score']}/100
- Risk level: {diag['risk_level']}
- Priority gaps: {", ".join(diag['gaps'])}

Personalized study plan
- Goal: {inputs.goal}
- Time: {inputs.hours_per_week}h/week for {inputs.weeks} weeks
- Strategy: rotate gap areas weekly; keep an error log; end each week with a short timed check.

Weekly goals
{chr(10).join(weekly)}

Recommended learning resources (URLs)
{res_md}

Progress feedback / next steps
- Track: weekly mini-test (30–45 min) + error log themes.
- If a gap stays <70 after 2 weeks, increase practice volume and add 1 extra review session.
"""

    return {
        "diagnosis": diag,
        "resources": resources,
        "report_markdown": llm_text.strip(),
        "llm_provider": llm_provider or "rule_based_fallback",
        "generated_on": date.today().isoformat(),
    }

