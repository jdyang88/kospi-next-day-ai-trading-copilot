APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
:root { --ink:#172033; --muted:#667085; --green:#087443; --green-soft:#eaf6f0; --line:#e6eaf0; --risk:#e24a4a; }
html, body, .stApp, [class*="css"] { font-family:'Noto Sans KR','Malgun Gothic',sans-serif; color:var(--ink); }
.stApp { background:#fbfcfe; }
[data-testid="stSidebar"] { background:#f7f9fb; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
.block-container { max-width:1440px; padding:1.7rem 2.4rem 3rem; }
h1 { font-size:1.72rem !important; letter-spacing:-0.035em; margin-bottom:.2rem !important; }
h2 { font-size:1.28rem !important; letter-spacing:-0.025em; }
h3 { font-size:1.02rem !important; }
.app-subtitle { color:var(--muted); font-size:.88rem; margin:.1rem 0 1.25rem; }
.status-line { border:1px solid var(--line); border-radius:10px; padding:.55rem .8rem; color:#475467; font-size:.82rem; background:white; }
.status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#16a061; margin-right:7px; }
.disclaimer { border-left:3px solid #8390a5; background:#f3f5f8; padding:.72rem .9rem; color:#536074; font-size:.8rem; margin:.7rem 0 1.2rem; }
.section-head { display:flex; align-items:baseline; justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:.65rem; margin:1.1rem 0 .5rem; }
.section-head strong { font-size:1.2rem; }
.section-head span { color:var(--muted); font-size:.76rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background:white; border-color:var(--line) !important; border-left:3px solid var(--green) !important; border-radius:10px !important; }
.rank { display:inline-flex; align-items:center; justify-content:center; width:27px; height:27px; border-radius:50%; background:var(--green); color:white; font-weight:700; font-size:.8rem; margin-right:.55rem; }
.stock-name { font-weight:700; font-size:.98rem; }
.ticker { color:var(--muted); font-size:.72rem; margin-left:.35rem; }
.prob { color:var(--green); font-size:1.18rem; font-weight:700; }
.label { color:var(--muted); font-size:.68rem; display:block; margin-bottom:.15rem; }
.value { color:var(--ink); font-size:.87rem; font-weight:600; }
.target { color:var(--green); }
.stop { color:var(--risk); }
.reason { color:#475467; font-size:.77rem; line-height:1.55; }
.confidence { color:var(--green); font-size:.72rem; font-weight:600; }
.decision { display:inline-block; margin:.38rem 0 0 2.45rem; padding:.16rem .48rem; border-radius:999px; font-size:.68rem; font-weight:700; }
.decision-buy { color:#05603a; background:#dff4e9; }
.decision-watch { color:#8a5a00; background:#fff3d6; }
.decision-hold { color:#7a3e3e; background:#fbe9e9; }
.decision-summary { border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; margin:.75rem 0 1rem; background:white; }
.decision-summary strong { display:block; font-size:1rem; margin-bottom:.2rem; }
.decision-summary span { color:var(--muted); font-size:.78rem; line-height:1.5; }
.metric-strip { background:white; border:1px solid var(--line); border-radius:11px; padding:.8rem .9rem; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:10px; padding:.75rem .85rem; }
[data-testid="stMetricLabel"] { color:var(--muted); }
.stButton > button, .stFormSubmitButton > button { border-radius:9px; font-weight:600; min-height:2.6rem; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:var(--green); border-color:var(--green); }
.stProgress > div > div > div > div { background-color:var(--green) !important; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.mode-note { color:#7a8699; font-size:.73rem; padding-top:.7rem; }
@media (max-width: 900px) { .block-container { padding:1rem; } .rec-row { padding:.75rem; } }
</style>
"""
