"""
Generate architecture diagram for Clinical AI Ops Agent.
Run: python docs/generate_diagram.py
Output: docs/architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")

# ── Color palette ──────────────────────────────────────────────
C_AGENT   = "#6366f1"   # indigo
C_AWS     = "#f59e0b"   # amber
C_DATA    = "#22c55e"   # green
C_ALERT   = "#ef4444"   # red
C_BG      = "#1e293b"   # dark blue-grey
C_TEXT    = "#f1f5f9"   # light
C_BORDER  = "#334155"

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.25):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=color, facecolor=C_BG,
    )
    ax.add_patch(rect)
    # Color accent bar on top
    accent = FancyBboxPatch(
        (x, y + h - 0.18), w, 0.18,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, alpha=0.85,
    )
    ax.add_patch(accent)
    ax.text(x + w/2, y + h/2 + 0.1, label,
            ha="center", va="center", color=C_TEXT,
            fontsize=9, fontweight="bold")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                ha="center", va="center", color="#94a3b8",
                fontsize=7)

def arrow(ax, x1, y1, x2, y2, label="", color="#64748b"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=1.5,
                    connectionstyle="arc3,rad=0.0",
                ))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my + 0.1, label,
                ha="center", va="center", color="#94a3b8", fontsize=7)

# ── Title ──────────────────────────────────────────────────────
ax.text(8, 9.55, "🏥  Clinical AI Ops Agent — Architecture",
        ha="center", va="center", color=C_TEXT,
        fontsize=14, fontweight="bold")
ax.text(8, 9.15, "Agentic ML Monitoring · AWS SageMaker · LangGraph ReAct",
        ha="center", va="center", color="#94a3b8", fontsize=9)

# ── EventBridge trigger ────────────────────────────────────────
box(ax, 0.4, 6.8, 2.2, 1.0, C_AWS, "EventBridge", "every 6 hours")

# ── Lambda agent ───────────────────────────────────────────────
box(ax, 3.2, 6.4, 2.8, 1.8, C_AGENT, "Agent Lambda", "LangGraph ReAct")

# ── Tool belt ──────────────────────────────────────────────────
box(ax, 0.4, 4.2, 2.2, 1.1, C_DATA,  "Drift Detector", "PSI + KS test")
box(ax, 3.2, 4.2, 2.2, 1.1, C_AWS,   "Retrain Trigger", "SageMaker Pipeline")
box(ax, 6.0, 4.2, 2.2, 1.1, C_AWS,   "Deploy Manager",  "Blue/Green")
box(ax, 8.8, 4.2, 2.2, 1.1, C_ALERT, "Alert Publisher", "SNS + Slack")

# ── AWS services ───────────────────────────────────────────────
box(ax, 0.4, 2.0, 2.2, 1.1, C_DATA, "S3 Data Lake", "logs · baselines · models")
box(ax, 3.2, 2.0, 2.2, 1.1, C_AWS,  "SageMaker", "Pipeline + Endpoint")
box(ax, 6.0, 2.0, 2.2, 1.1, C_AWS,  "DynamoDB", "Agent State")
box(ax, 8.8, 2.0, 2.2, 1.1, C_ALERT,"SNS Topic", "Email · PagerDuty")
box(ax, 11.6, 2.0, 2.2, 1.1, C_DATA, "CloudWatch", "Metrics + Alarms")

# ── Amazon Bedrock ─────────────────────────────────────────────
box(ax, 11.6, 6.4, 2.2, 1.8, C_AGENT, "Bedrock LLM", "Claude 3 Sonnet")

# ── Arrows: EventBridge → Lambda ──────────────────────────────
arrow(ax, 2.6, 7.3, 3.2, 7.3, "trigger", C_AWS)

# ── Arrows: Lambda ↔ Bedrock ──────────────────────────────────
arrow(ax, 6.0, 7.3, 11.6, 7.3, "reason", C_AGENT)
arrow(ax, 11.6, 7.0, 6.0, 7.0, "response", C_AGENT)

# ── Arrows: Lambda → Tools ────────────────────────────────────
arrow(ax, 4.6, 6.4, 1.5, 5.3,  "", C_AGENT)
arrow(ax, 4.6, 6.4, 4.3, 5.3,  "", C_AGENT)
arrow(ax, 4.6, 6.4, 7.1, 5.3,  "", C_AGENT)
arrow(ax, 4.6, 6.4, 9.9, 5.3,  "", C_AGENT)

# ── Arrows: Tools → AWS Services ──────────────────────────────
arrow(ax, 1.5, 4.2, 1.5, 3.1,  "", C_DATA)
arrow(ax, 4.3, 4.2, 4.3, 3.1,  "", C_AWS)
arrow(ax, 7.1, 4.2, 7.1, 3.1,  "", C_AWS)
arrow(ax, 9.9, 4.2, 9.9, 3.1,  "", C_ALERT)
arrow(ax, 4.6, 6.5, 7.1, 3.1,  "", "#64748b")   # Lambda → DynamoDB
arrow(ax, 4.6, 6.5, 12.7, 3.1, "", "#64748b")   # Lambda → CloudWatch

# ── Legend ─────────────────────────────────────────────────────
legend_items = [
    (C_AGENT, "Agent / LLM"),
    (C_AWS,   "AWS Services"),
    (C_DATA,  "Data / Storage"),
    (C_ALERT, "Alerts"),
]
for i, (color, label) in enumerate(legend_items):
    rx = 0.5 + i * 2.8
    ry = 0.4
    rect = FancyBboxPatch((rx, ry), 0.35, 0.35,
                           boxstyle="round,pad=0,rounding_size=0.05",
                           facecolor=color, linewidth=0)
    ax.add_patch(rect)
    ax.text(rx + 0.5, ry + 0.17, label,
            va="center", color=C_TEXT, fontsize=8)

# ── Flow label ─────────────────────────────────────────────────
ax.text(8, 0.2,
        "Flow: Schedule → Detect Drift → Retrain (if needed) → Validate → Deploy → Alert",
        ha="center", va="center", color="#64748b", fontsize=8, style="italic")

plt.tight_layout(pad=0.5)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved docs/architecture.png")
plt.show()


if __name__ == "__main__":
    pass  # runs on import from generate_diagram.py
