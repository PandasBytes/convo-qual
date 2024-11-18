import plotly.express as px
import pandas as pd

LOGO_PATH = '/Users/RiRi/Desktop/github/convo-quality/src/LLM_classification_pipeline/valence.png'
# Dummy Data with richer sub-subtopics
data = pd.DataFrame({
    "topics": [
        "Leadership Development", "Leadership Development", "Leadership Development",
        "Leadership Development", "Leadership Development",
        "Career Growth", "Career Growth", "Career Growth", "Career Growth",
        "Career Growth",
        "Conflict Resolution", "Conflict Resolution", "Conflict Resolution",
        "Conflict Resolution", "Conflict Resolution",
        "Productivity", "Productivity", "Productivity", "Productivity",
        "Productivity",
        "Strategic Thinking", "Strategic Thinking", "Strategic Thinking", "Strategic Thinking"
    ],
    "subtopics": [
        "Executive Presence", "Executive Presence", "Team Building",
        "Motivational Leadership", "Motivational Leadership",
        "Networking", "Networking", "Resume Building", "Career Planning",
        "Career Planning",
        "Managing Difficult Conversations", "Managing Difficult Conversations",
        "Workplace Mediation", "Conflict Strategies", "Conflict Strategies",
        "Time Management", "Time Management", "Focus Strategies",
        "Task Automation", "Task Automation",
        "Decision Making", "Decision Making", "Long-Term Planning", "Long-Term Planning"
    ],
    "sub_subtopics": [
        "Body Language", "Confidence Building", "Team Dynamics",
        "Inspiring Vision", "Delegation",
        "LinkedIn Networking", "Personal Branding", "CV Formatting", "Setting Goals",
        "Career Pathways",
        "Emotional Regulation", "Listening Skills", "Negotiation Skills",
        "Resolution Frameworks", "Conflict Mapping",
        "Pomodoro Technique", "Daily Planning", "Minimizing Distractions",
        "Workflow Optimization", "Automated Tools",
        "Scenario Analysis", "Risk Evaluation", "Goal Roadmaps", "Vision Development"
    ],
    "values": [
        15, 15, 25, 20, 10, 10, 8, 12, 18, 10, 10, 10, 15, 15, 10,
        12, 13, 10, 15, 15, 12, 13, 15, 10
    ]
})

# Create the Sunburst chart with hierarchy
fig = px.sunburst(
    data,
    path=["topics", "subtopics", "sub_subtopics"],  # Hierarchy with sub-subtopics
    values="values",  # Size of each slice
    title="Valence Career Coaching Topics - FAKE DATA",
    color="topics",  # Color by main topic
    color_discrete_map={
        "Leadership Development": "#4B75E0",  # calm, confident blue
        "Career Growth": "#E96A8D",           # vibrant, aspirational deep pink
        "Conflict Resolution": "#80C2FF",     # soothing, trust-building light blue
        "Productivity": "#FFD1E5",            # motivating, energetic pink
        "Strategic Thinking": "#375AB9"       # focused, strategic navy blue
    }
)

# Set the initial view to show topics only
fig.update_traces(
    level="topics",  # Start with only the topics visible
    textinfo="label+percent parent",  # Show labels and percentages
    hoverinfo="none"  # No hover popups
)

# Update layout for better interaction
fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),  # Adjust margins
)

# Add annotations for insights
savings = "$8,888"
fig.add_annotation(
    x=1,
    y=0.9,
    text=f"<b>{savings}</b><br><span style='font-size:12px;'>TOTAL SAVINGS</span>",
    showarrow=False,
    font=dict(size=24, color="black"),
    align="right"
)
conv_count = "700"
fig.add_annotation(
    x=1,
    y=0.8,
    text=f"<b>{conv_count}</b><br><span style='font-size:12px;'>TOTAL COACHING SESSIONS</span>",
    showarrow=False,
    font=dict(size=24, color="black"),
    align="right"
)

fig.add_annotation(
    x=1,
    y=0.6,
    text=f"<b>USAGE STATS</b><br><span style='font-size:12px;'>",
    showarrow=False,
    font=dict(size=20, color="black"),
    align="right"
)

avg_rating = 3.3
fig.add_annotation(
    x=1,
    y=0.45,
    text=f"<b>{avg_rating}</b><br><span style='font-size:12px;'>Avg Satisfaction Rating</span>",
    showarrow=False,
    font=dict(size=24, color="black"),
    align="right"
)

duration = "5.2 min"
fig.add_annotation(
    x=1,
    y=0.32,
    text=f"<b>{duration}</b><br><span style='font-size:12px;'> Avg Convo Duration</span>",
    showarrow=False,
    font=dict(size=24, color="black"),
    align="right"
)

follow_up = "72%"
fig.add_annotation(
    x=1,
    y=0.22,
    text=f"<b>{follow_up}</b><br><span style='font-size:12px;'> Convo Follow Up Rate</span>",
    showarrow=False,
    font=dict(size=24, color="black"),
    align="right"
)

fig.update_traces(hovertemplate="")  # Remove hover annotations

fig.show()
fig.write_html("/Users/RiRi/Desktop/github/convo-quality/viz/career_coaching_burst.html")
