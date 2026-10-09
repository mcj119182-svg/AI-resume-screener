"""AI 智能简历筛选器 —— 前端界面

视觉设计要点
------------
* 深色主题 + 渐变强调色，卡片化布局，信息层级清晰
* 匹配度用环形进度（SVG）呈现，比数字或进度条更直观
* 技能用胶囊标签（pill）展示，替代逐条 st.badge
* 所有动态内容（AI 返回的姓名 / 点评 / 技能、数据库字段）一律 html.escape，
  因为本页用 unsafe_allow_html=True 渲染，不转义就有存储型 XSS 风险
"""
import html
import os
from contextlib import closing

import pandas as pd
import pymysql
import streamlit as st
from dotenv import load_dotenv

from ai_screener import ask_ai_to_screen, save_to_database, validate_ai_payload
from recruit_db_helper import get_recruit_connection

# 访问密码从 .env 的 WEB_PASSWORD 读取，不再硬编码
load_dotenv()
WEB_PASSWORD = os.getenv("WEB_PASSWORD")

# 未配置密码或密码不匹配都拒绝访问
password = st.text_input("请输入访问密码", type="password")
if not WEB_PASSWORD or password != WEB_PASSWORD:
    st.stop()

# ============ 页面基础配置 ============
st.set_page_config(
    page_title="AI 智能简历筛选器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 主题 CSS ============
THEME_CSS = """
<style>
:root {
    --bg:        #0a0f1a;
    --panel:     #131a2a;
    --panel-2:   #1a2338;
    --line:      #24304a;
    --text:      #e6ebf5;
    --muted:     #8a95ad;
    --brand-1:   #6366f1;
    --brand-2:   #a855f7;
    --brand-3:   #ec4899;
    --ok:        #22c55e;
    --warn:      #f59e0b;
    --bad:       #ef4444;
    --radius:    16px;
}

/* ---------- 全局 ---------- */
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #16203a 0%, var(--bg) 55%) no-repeat; }
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
                 "PingFang SC", "Hiragino Sans GB", sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- 标题 ---------- */
.hero-title {
    font-size: 2.35rem; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(100deg, var(--brand-1) 0%, var(--brand-2) 48%, var(--brand-3) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .35rem 0; line-height: 1.25;
}
.hero-sub { color: var(--muted); font-size: 1rem; margin-bottom: 1.6rem; }
.section-head {
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.06rem; font-weight: 700; color: var(--text);
    margin: 1.6rem 0 .9rem 0;
}
.section-head::before {
    content: ""; width: 4px; height: 1.05em; border-radius: 3px;
    background: linear-gradient(180deg, var(--brand-1), var(--brand-2));
}

/* ---------- 卡片 ---------- */
.c-card {
    background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line); border-radius: var(--radius);
    padding: 1.15rem 1.35rem; box-shadow: 0 6px 22px rgba(0,0,0,.28);
    height: 100%;
}
.c-card.accent-review { border-left: 3px solid var(--brand-2); }
.c-card.accent-ok     { border-left: 3px solid var(--ok); }

.c-label {
    color: var(--muted); font-size: .78rem; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: .55rem;
}
.c-text { color: var(--text); font-size: 1.02rem; line-height: 1.75; }
.c-text-sm { color: var(--text); font-size: .92rem; line-height: 1.6; }

/* ---------- 指标卡 ---------- */
.mstat {
    background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.1rem; text-align: center;
}
.mstat-val { font-size: 1.7rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.mstat-lab { font-size: .8rem; color: var(--muted); margin-top: .3rem; }
.mstat-val.is-ok   { color: var(--ok); }
.mstat-val.is-warn { color: var(--warn); }
.mstat-val.is-brand{ color: #b9a6ff; }

/* ---------- 匹配度环 ---------- */
.ring-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring { width: 148px; height: 148px; }
.ring-num { font-size: 34px; font-weight: 800; fill: #e6ebf5; }
.ring-unit { font-size: 12px; fill: #8a95ad; }
.ring-cap {
    margin-top: .5rem; font-size: .9rem; font-weight: 700;
    padding: .22rem .8rem; border-radius: 999px;
}

/* ---------- 技能胶囊 ---------- */
.pills { display: flex; flex-wrap: wrap; gap: .45rem; }
.pill {
    background: rgba(99,102,241,.16); color: #c3bdff;
    border: 1px solid rgba(99,102,241,.38); border-radius: 999px;
    padding: .3rem .78rem; font-size: .84rem; font-weight: 600; white-space: nowrap;
}

/* ---------- 候选人名 ---------- */
.who { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
.who-name { font-size: 1.5rem; font-weight: 800; color: var(--text); }
.who-meta { color: var(--muted); font-size: .85rem; }

/* ---------- 空状态 ---------- */
.empty {
    text-align: center; padding: 3.2rem 1rem; border: 1px dashed var(--line);
    border-radius: var(--radius); background: rgba(19,26,42,.55);
}
.empty-icon { font-size: 2.6rem; }
.empty-title { color: var(--text); font-weight: 700; margin-top: .6rem; }
.empty-sub { color: var(--muted); font-size: .9rem; margin-top: .25rem; }

/* ---------- Streamlit 组件微调 ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important; font-weight: 700 !important;
    border: 1px solid var(--line) !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 20px rgba(99,102,241,.28);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, var(--brand-1), var(--brand-2)) !important;
    border: none !important; color: #fff !important;
}
.stTextArea textarea, .stTextInput input {
    border-radius: 12px !important;
    background: rgba(19,26,42,.85) !important;
    border: 1px solid var(--line) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--brand-1) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.18) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(19,26,42,.7); border: 1px dashed #33415f;
    border-radius: var(--radius); padding: .9rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--brand-1); }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #111a2e 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] .stRadio label {
    padding: .5rem .6rem; border-radius: 10px; transition: background .15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(99,102,241,.12); }

/* 分段控件 / 标签页 */
[data-testid="stSegmentedControl"] button { border-radius: 10px !important; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: .5rem 1rem; font-weight: 600;
}

/* 表格 */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line) !important; border-radius: 12px; overflow: hidden;
}
table tr:hover td { background: rgba(99,102,241,.08); }

/* 提示条 */
[data-testid="stAlert"] { border-radius: 12px; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============ 通用小组件 ============
def esc(value):
    """所有进入 HTML 的动态内容都必须过这一层。"""
    return html.escape(str(value if value is not None else ""))


def section(title):
    st.markdown("<div class='section-head'>%s</div>" % esc(title), unsafe_allow_html=True)


def stat_card(label, value, tone=""):
    cls = " is-%s" % tone if tone else ""
    return ("<div class='mstat'><div class='mstat-val%s'>%s</div>"
            "<div class='mstat-lab'>%s</div></div>"
            % (cls, esc(value), esc(label)))


def score_meta(score):
    """分数 → (主色, 文案)"""
    if score is None:
        return "#8a95ad", "未评分"
    if score >= 80:
        return "#22c55e", "强烈推荐"
    if score >= 60:
        return "#f59e0b", "可以聊聊"
    return "#ef4444", "暂不匹配"


def score_ring(score):
    """环形匹配度：SVG 画圆环，避免依赖任何图表库。"""
    if score is None:
        score = 0
    score = max(0, min(int(score), 100))
    color, caption = score_meta(score)
    radius, stroke = 62, 11
    circumference = 2 * 3.141592653589793 * radius
    dash = circumference * score / 100
    gradient_id = "ringGrad%d" % score
    return """
<div class="ring-wrap">
  <svg class="ring" viewBox="0 0 160 160">
    <defs>
      <linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#6366f1"/>
        <stop offset="100%" stop-color="{color}"/>
      </linearGradient>
    </defs>
    <circle cx="80" cy="80" r="{r}" fill="none" stroke="#24304a" stroke-width="{sw}"/>
    <circle cx="80" cy="80" r="{r}" fill="none" stroke="url(#{gid})" stroke-width="{sw}"
            stroke-linecap="round" stroke-dasharray="{dash} {rest}"
            transform="rotate(-90 80 80)"/>
    <text class="ring-num" x="80" y="76" text-anchor="middle"
          dominant-baseline="central">{score}</text>
    <text class="ring-unit" x="80" y="106" text-anchor="middle">/ 100 分</text>
  </svg>
  <div class="ring-cap" style="color:{color};background:{color}1f;border:1px solid {color}55">
    {caption}
  </div>
</div>
""".format(gid=gradient_id, color=color, r=radius, sw=stroke,
           dash=round(dash, 2), rest=round(circumference - dash, 2),
           score=score, caption=esc(caption))


def skill_list_of(skills):
    """兼容 list / 字符串（含全角逗号、顿号）两种形式。"""
    if isinstance(skills, list):
        items = [str(s).strip() for s in skills]
    else:
        text = str(skills or "").replace("，", ",").replace("、", ",").replace("；", ",")
        items = [s.strip() for s in text.replace(";", ",").split(",")]
    return [s for s in items if s]


def pill_row(items):
    return "<div class='pills'>%s</div>" % "".join(
        "<span class='pill'>%s</span>" % esc(s) for s in items
    )


def empty_state(icon, title, sub=""):
    st.markdown(
        "<div class='empty'><div class='empty-icon'>%s</div>"
        "<div class='empty-title'>%s</div><div class='empty-sub'>%s</div></div>"
        % (esc(icon), esc(title), esc(sub)),
        unsafe_allow_html=True,
    )


def fetch_candidates(limit):
    """按 limit 拉取候选人；时间列名不同的老库自动降级。"""
    conn = None
    try:
        conn = get_recruit_connection()
        with closing(conn.cursor(pymysql.cursors.DictCursor)) as cursor:
            try:
                cursor.execute(
                    "SELECT id, name, skills, match_score, ai_review, resume_file_path, create_time "
                    "FROM candidates ORDER BY id DESC LIMIT %s", (limit,))
            except pymysql.err.MySQLError:
                # 老库可能没有 create_time 列
                cursor.execute(
                    "SELECT id, name, skills, match_score, ai_review, resume_file_path "
                    "FROM candidates ORDER BY id DESC LIMIT %s", (limit,))
            return cursor.fetchall()
    finally:
        if conn is not None:
            conn.close()


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown(
        "<div style='padding:.4rem 0 1rem 0'>"
        "<div style='font-size:1.35rem;font-weight:800;color:#e6ebf5'>🤖 智能招聘助手</div>"
        "<div style='color:#8a95ad;font-size:.82rem;margin-top:.2rem'>Resume Screener Agent</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    menu = ["📄 简历筛选", "📊 历史记录"]
    choice = st.radio("选择功能", menu, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div style='color:#8a95ad;font-size:.8rem;line-height:1.9'>"
        "模型 · DeepSeek<br>调度 · Function Calling<br>阈值 · ≥80 分推荐面试"
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 历史记录页
# =====================================================================
if choice == "📊 历史记录":
    st.markdown("<div class='hero-title'>历史候选人记录</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>所有经过 AI 评估的候选人，支持搜索、筛选与导出。</div>",
                unsafe_allow_html=True)

    try:
        results = fetch_candidates(500)
    except Exception as e:
        st.error("读取数据库失败：%s" % e)
        st.stop()

    if not results:
        empty_state("📭", "暂无历史数据", "去「简历筛选」上传第一份简历试试吧")
        st.stop()

    df = pd.DataFrame(results)
    if "match_score" in df.columns:
        df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce").fillna(0).astype(int)
    else:
        df["match_score"] = 0

    # ---- 统计卡 ----
    total = len(df)
    scores = df["match_score"].dropna()
    avg = float(scores.mean()) if len(scores) else 0.0
    top = int(scores.max()) if len(scores) else 0
    high = int((scores >= 80).sum())
    cols = st.columns(4)
    for col, (label, value, tone) in zip(cols, [
        ("候选人总数", total, ""),
        ("平均匹配度", "%.1f 分" % avg, "is-brand"),
        ("最高匹配度", "%d 分" % top, "is-ok"),
        ("80 分以上", high, "is-warn"),
    ]):
        with col:
            st.markdown(stat_card(label, value, tone), unsafe_allow_html=True)

    # ---- 筛选 ----
    section("候选人列表")
    f1, f2, f3 = st.columns([2, 1.2, 1.2])
    with f1:
        keyword = st.text_input("搜索", placeholder="按姓名或技能搜索…",
                                label_visibility="collapsed")
    with f2:
        band = st.segmented_control(
            "区间", ["全部", "80+", "60-79", "<60"], default="全部",
            label_visibility="collapsed",
        ) or "全部"
    with f3:
        sort_by = st.segmented_control(
            "排序", ["分数↓", "最近"], default="分数↓", label_visibility="collapsed",
        ) or "分数↓"

    view = df.copy()
    if keyword.strip():
        kw = keyword.strip().lower()
        mask = (view["name"].astype(str).str.lower().str.contains(kw, regex=False)
                | view["skills"].astype(str).str.lower().str.contains(kw, regex=False))
        view = view[mask]
    if band == "80+":
        view = view[view["match_score"] >= 80]
    elif band == "60-79":
        view = view[(view["match_score"] >= 60) & (view["match_score"] < 80)]
    elif band == "<60":
        view = view[view["match_score"] < 60]
    if sort_by == "最近" and "create_time" in view.columns:
        view = view.sort_values("create_time", ascending=False)
    else:
        view = view.sort_values("match_score", ascending=False)

    if view.empty:
        empty_state("🔍", "没有符合条件的候选人", "换个关键词或切换分数区间试试")
    else:
        st.caption("共 %d 条记录" % len(view))
        show_cols = [c for c in ["name", "skills", "match_score", "ai_review",
                                 "resume_file_path", "create_time"] if c in view.columns]
        st.dataframe(
            view[show_cols],
            use_container_width=True,
            hide_index=True,
            height=min(560, 96 + 36 * len(view)),
            column_config={
                "name": st.column_config.TextColumn("姓名", width="small"),
                "skills": st.column_config.TextColumn("核心技能", width="medium"),
                "match_score": st.column_config.ProgressColumn(
                    "匹配度", min_value=0, max_value=100, format="%d 分", width="medium",
                ),
                "ai_review": st.column_config.TextColumn("AI 点评", width="large"),
                "resume_file_path": st.column_config.TextColumn("来源", width="small"),
                "create_time": st.column_config.TextColumn("评估时间", width="small"),
            },
        )

    st.download_button(
        "⬇️ 导出全部记录（CSV）",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="候选人历史记录.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.stop()  # 选了历史记录就停止执行后面的代码


# =====================================================================
# 简历筛选页
# =====================================================================
st.markdown("<div class='hero-title'>AI 智能简历筛选器</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>上传或粘贴简历，Agent 会先查重、再提取技能、最后给出匹配度与点评。</div>",
    unsafe_allow_html=True,
)

section("① 提供简历")
mode = st.radio("输入方式", ["📤 上传文件", "✍️ 粘贴文本"],
                horizontal=True, label_visibility="collapsed")

resume_text = ""
if mode == "📤 上传文件":
    uploaded_file = st.file_uploader("拖拽或点击上传 .txt 简历", type=["txt"])
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8", errors="replace")
        st.caption("✅ 已读取 %s（%d 字符）" % (uploaded_file.name, len(resume_text)))
else:
    pasted = st.text_area(
        "粘贴简历文本",
        height=220,
        placeholder="例如：我叫李四，熟练掌握 Python 和 MySQL，做过两个 AI 项目…",
        label_visibility="collapsed",
    )
    if pasted.strip():
        resume_text = pasted

if resume_text.strip():
    with st.expander("👀 简历预览（%d 字符）" % len(resume_text), expanded=False):
        st.text(resume_text)

section("② 开始分析")
analyze = st.button("🚀 开始 AI 分析", type="primary", use_container_width=True)

if analyze:
    if not resume_text.strip():
        st.warning("请先上传文件或粘贴简历内容。")
        st.stop()

    ai_result_str = None
    ai_data = None
    save_status = None
    v_status = None

    # spinner 只包裹 AI 调用与入库，避免 st.stop() 在 spinner 内部中断导致转圈不停
    with st.spinner("🧠 Agent 正在分析：查重 → 提取技能 → 打分…"):
        ai_result_str = ask_ai_to_screen(resume_text)
        if ai_result_str is not None:
            # 后端统一校验并规范化字段，前端不再自己解析 JSON
            v_status, ai_data = validate_ai_payload(ai_result_str, resume_text)
            if v_status == "ok":
                save_status, _ = save_to_database(ai_data, "Streamlit网页调用")

    if ai_result_str is None:
        st.error("AI 请求失败，请检查网络或 API Key 后重试。")
        st.stop()

    # ---- 分支一：命中查重规则，Agent 自动跳过 ----
    if v_status == "skipped":
        skipped_name = (ai_data or {}).get("name", "该候选人")
        reason = (ai_data or {}).get("ai_review", "")
        st.markdown(
            "<div class='c-card' style='border-left:3px solid #f59e0b'>"
            "<div class='c-label'>⏭️ 已自动跳过重复评估</div>"
            "<div class='c-text-sm'>%s</div></div>" % esc(reason),
            unsafe_allow_html=True,
        )
        st.info("本次未重新打分、未写入数据库（查重规则：最近 3 天内已评估且分数高于 80 分）。")
        st.stop()

    # ---- 分支二：不像有效简历 ----
    if ai_data is None:
        st.warning("⚠️ 这份内容不像一份有效简历（可能是乱码、空文本或无关内容），未保存。")
        st.stop()

    # ---- 分支三：入库失败 ----
    if save_status != "ok":
        st.error("❌ 分析完成，但写入数据库失败，请检查数据库连接后重试。")
        st.stop()

    st.success("🎉 分析完成，结果已存入数据库。")

    # ---- 结果展示 ----
    name = ai_data["name"]
    skills = ai_data["skills"]
    match_score = ai_data["match_score"]
    ai_review = ai_data["ai_review"]
    color, _ = score_meta(match_score)

    section("③ 评估结果")
    st.markdown(
        "<div class='who'><span class='who-name'>👤 %s</span>"
        "<span class='who-meta'>来源：Streamlit 网页</span></div>" % esc(name),
        unsafe_allow_html=True,
    )
    st.write("")

    col_ring, col_detail = st.columns([1, 2.4])
    with col_ring:
        st.markdown("<div class='c-card'>%s</div>" % score_ring(match_score),
                    unsafe_allow_html=True)
    with col_detail:
        st.markdown(
            "<div class='c-card accent-review' style='margin-bottom:1rem'>"
            "<div class='c-label'>💬 AI 点评</div>"
            "<div class='c-text'>%s</div></div>" % esc(ai_review),
            unsafe_allow_html=True,
        )
        items = skill_list_of(skills)
        if items:
            st.markdown(
                "<div class='c-card'><div class='c-label'>🛠 核心技能（%d 项）</div>%s</div>"
                % (len(items), pill_row(items)),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='c-card'><div class='c-label'>🛠 核心技能</div>"
                "<div class='c-text-sm' style='color:#8a95ad'>未提取到明确技能</div></div>",
                unsafe_allow_html=True,
            )

    with st.expander("🔍 查看原始数据（JSON）"):
        st.json(ai_data)
