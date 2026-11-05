"""Two-page PDF submission report (Chinese), built from the model outputs.

    python src/make_report.py   ->  outputs/特斯拉超充选址_提交报告.pdf
needs: outputs/final_10_sites.csv, outputs/fig_overview.svg (make_figures.py), outputs/fig_webmap.jpg (web screenshot)
"""
import base64, html
import pandas as pd
from config import OUT

T = pd.read_csv(OUT / "final_10_sites.csv")
SVG = (OUT / "fig_overview.svg").read_text(encoding="utf-8")
WEB = base64.b64encode((OUT / "fig_webmap.jpg").read_bytes()).decode()
CITY_CN = {"86283631fffffff": "斯托克顿", "8629a0c47ffffff": "温彻斯特", "8629b6d37ffffff": "菲尼克斯",
           "86268c60fffffff": "科罗拉多泉", "862a100afffffff": "纽约布朗克斯", "86441a597ffffff": "奥本代尔",
           "8628f0137ffffff": "俄勒冈城", "8629a970fffffff": "洛斯巴诺斯", "862ab2ce7ffffff": "底特律", "862a94d07ffffff": "托莱多"}
e = html.escape


def rows():
    out = []
    for _, r in T.iterrows():
        host = r["宿主"].split(" (")[0]
        stalls = f'{r["超充桩(V4)"]}' + (f'<span class="m">（{r["其中拖挂直通车位"]} 拖挂）</span>' if r["其中拖挂直通车位"] else "")
        ic = r["并网方式"].split("（")[0].replace("灵活接入", "灵活")
        seg = '<span class="cor">走廊</span>' if r["类型"] == "高速走廊" else "都市圈"
        out.append(f'<tr><td class="c">{r["排名"]}</td><td><b>{CITY_CN[r["H3_res6"]]}</b>, {r["州"]}<br><span class="m">{seg}</span></td>'
                   f'<td>{e(host)}<br><span class="m">{e(r["地址"])}</span></td>'
                   f'<td class="n">{r["纬度"]:.5f},<br>{r["经度"]:.5f}</td><td class="c">{stalls}</td><td class="c">{r["慢充(目的地L2)"]}</td>'
                   f'<td>{ic}<br><span class="m">{r["申请功率_kW"]:,} kW · {r["预计通电_月"]} 个月</span></td>'
                   f'<td class="n">{r["首年服务电量_MWh"]:,}</td><td class="n">{r["NPV_万美元"] / 100:.1f}</td></tr>')
    return "\n".join(out)


HTML = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>
@page{{size:A4;margin:10mm 11mm 9mm}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Noto Sans CJK SC","Noto Sans SC",sans-serif;font-size:8.6pt;line-height:1.5;color:#1b2230}}
h1{{font-size:15.5pt;margin:0;letter-spacing:.3px}}
.top{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #1d3a6e;padding-bottom:4px;margin-bottom:6px}}
.top .by{{color:#5b6576;font-size:8pt;text-align:right}}
h2{{font-size:10.6pt;margin:7px 0 3px;color:#1d3a6e;display:flex;align-items:center;gap:6px}}
h2 i{{font-style:normal;background:#1d3a6e;color:#fff;border-radius:3px;font-size:8.5pt;padding:0 5px}}
p{{margin:2px 0 3px}}
.lead{{background:#eef3fb;border-left:3px solid #2f6fdf;padding:4px 8px;margin:4px 0 6px}}
figure{{margin:0}}
figcaption{{font-size:7.4pt;color:#5b6576;margin-top:2px}}
.fig1 svg{{width:100%;height:auto;display:block;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:7.6pt;line-height:1.3}}
th{{background:#1d3a6e;color:#fff;font-weight:500;padding:3px 4px;text-align:left}}
td{{padding:1.6px 4px;border-bottom:1px solid #e1e6ee;vertical-align:top}}
tr:nth-child(even) td{{background:#f7f9fc}}
td.c{{text-align:center}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.m{{color:#6a7385;font-size:7pt}} .cor{{color:#b86a14;font-weight:700}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;margin-top:4px}}
.box{{border:1px solid #d9e0ea;border-radius:5px;padding:4px 7px}}
.box h3{{margin:0 0 2px;font-size:9pt;color:#1d3a6e}}
ul{{margin:0;padding-left:13px}} li{{margin:1px 0}}
b.k{{color:#1d3a6e}}
.page2{{break-before:page}}
.two{{display:grid;grid-template-columns:1.05fr .95fr;gap:10px}}
.chain{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:3px 0 2px}}
.step{{background:#f3f6fb;border:1px solid #d9e0ea;border-radius:4px;padding:3px 5px;font-size:7.5pt;line-height:1.35}}
.step b{{display:block;color:#1d3a6e;font-size:7.9pt}}
.mt td,.mt th{{font-size:7.5pt;padding:2px 4px}}
.fig2 img{{width:100%;border-radius:5px;display:block}}
.note{{color:#5b6576;font-size:7.4pt}}
</style></head><body>

<div class="top"><h1>在美国给特斯拉超充选址：10 个推荐站点与方法说明</h1>
<div class="by">提交材料 · 2026 年 11 月<br>代码、数据与交互地图均可复现</div></div>

<div class="lead">我把这个问题当成一个<b>多目标、带电网约束的设施选址问题</b>来做：用 Uber H3 网格统一全美空间，
用特斯拉 2016–2025 年 <b>4,345 个真实站点</b>回测选址信号，再把“选哪里、建多大、怎么并网”放进同一个整数规划里一起优化，最后逐个落到真实宿主地址。
结果是 10 个站、<b>204 个 V4 超充桩 + 76 个目的地慢充</b>，组合 NPV 约 <b>$64.0M</b>（模型估算，CAPEX $12.5M）。</div>

<h2><i>1</i>10 个推荐地址</h2>
<figure class="fig1">{SVG}<figcaption>图 1　10 个推荐站点（蓝：都市圈站，橙：I-5 走廊站）与特斯拉现有网络（灰点：已开通；黄点：在建/许可/规划，共 4,268 站）。
所有宿主距任何已开通或规划中的特斯拉站 ≥ 5.6 km。</figcaption></figure>

<table style="margin-top:5px"><tr><th>#</th><th>城市</th><th>宿主 / 详细地址</th><th>纬度, 经度</th><th>V4 超充</th><th>L2 慢充</th><th>并网方式</th><th>首年 MWh</th><th>NPV $M</th></tr>
{rows()}</table>

<div class="grid3">
<div class="box"><h3>运营方式</h3><ul>
<li>对所有 NACS 车辆开放，特斯拉 / 非特斯拉、会员 / 非会员差异定价</li>
<li>分时电价引导错峰；占用率 &gt; 80% 启用拥堵费，充满后收占位费</li>
<li>动态灵活接入站按电力公司逐时功率包络运行，限电时<b>降单桩功率而不关桩</b>，优先低电量车辆</li>
<li>走廊站设 4 个拖挂直通位，节假日高峰启用预约与 80% SOC 上限</li></ul></div>
<div class="box"><h3>经济增效</h3><ul>
<li><b class="k">灵活接入</b>：通电时间从常规的 18–30 个月缩到约 8 个月；与全部走常规接入相比，组合 NPV 高约 <b>29%</b>（$64.0M 对 $49.6M）</li>
<li><b class="k">按 NPV 定规模</b>：16–32 桩，而不是一律做大；基准情景下 0 台 Megapack，只在馈线余量不足时再加</li>
<li><b class="k">帕累托取舍</b>：比纯利润最优少 9% NPV，换来走廊服务电量翻倍（4.6 → 9.3 GWh/年）</li></ul></div>
<div class="box"><h3>方案亮点</h3><ul>
<li>用特斯拉十年真实开站做回测和前向验证，而不是拍脑袋定权重</li>
<li>选址、规模、并网方式联合优化，输出的是可执行配置</li>
<li>把电网从“硬约束”变成可选项（灵活接入包络 + 储能）</li>
<li>H3 作为全流程空间底座；每个网格都核到真实宿主，不合格就剔除重解</li></ul></div>
</div>

<div class="page2">
<div class="two">
<div>
<h2><i>2</i>选址逻辑与迭代链条</h2>
<p><b>目标</b>：① 路径可达与流量覆盖；② 降低里程焦虑；③ 投资回报与电网匹配（权重最高，2/3）；④ 少与竞品和特斯拉自有站互相分流。
<b>约束</b>：电网容量与并网周期、土地与宿主可用性、交通流 / 人口 / 出行行为。</p>
<p><b>数据</b>：supercharge.info 全量特斯拉站（开通日期、桩数、状态）、美国能源部 AFDC 竞品快充 12,558 站（2019/2022/2025 快照）、县人口与各州电动车保有量、
Natural Earth 公路网、Google Places 宿主信息。</p>
<p><b>流程</b>：H3 底座（res-8 路网 → res-6 候选 51,653 格 → res-4 OD 区）→ 走廊用 7,000 条重力 OD 路径上的加油流模型（FRLM），
本地用 MCI-Huff 引力模型（计入竞品、拥堵、自我分流）→ 回测校准的筛选打分 → 159 个候选共生成 584 个“桩数 × 并网方式 × 储能”配置列，逐时运营模型算每列 NPV →
整数规划选 10 站（站距 ≥ 40 km、每州 ≤ 3、距特斯拉站 ≥ 6 km、走廊电量 ε-约束）→ 宿主尽调。</p>
<div class="chain">
<div class="step"><b>① 搭框架</b>梳理学术与商业前沿，定 4 目标 3 约束</div>
<div class="step"><b>② v0 单模型</b>都市圈≈人口基线，走廊几乎找不到</div>
<div class="step"><b>③ 分段 + 滚动校准</b>都市圈 / 走廊分开打分，每年重校准</div>
<div class="step"><b>④ 经济性修正</b>吸引力饱和、区域电价、州上限，解决桩数顶格、加州扎堆</div>
<div class="step"><b>⑤ 尽调回路</b>北拉斯维加斯网格贴着内利斯空军基地 → 剔除重解</div>
<div class="step"><b>⑥ 对照启发式</b>“需求/供给^0.5”AUC 仅 0.63，选点落到海上、山顶</div>
<div class="step"><b>⑦ 学习型筛选</b>14 特征逻辑回归，都市圈 AUC 0.757 → 0.836</div>
<div class="step"><b>⑧ v2 重跑</b>换 4 站，走廊电量 +70%，前向验证 86% → 88%</div>
</div>

<h2><i>3</i>AI 解题思路与人机分工</h2>
<p><b>核心拆解</b>：把“选 10 个地址”拆成 <b>候选空间（H3）→ 需求与竞争（FRLM + Huff）→ 历史回测（特斯拉怎么选）→ 逐站配置与经济性 → 选址-配置联合优化（MILP）→ 落地尽调</b>，
每一步都有可检验的中间产物。</p>
<p><b>分工</b>：我负责定义问题、设目标权重和业务规则（ROI 权重最高、每州上限、新站距离），判断每一轮结果是否合理并决定下一步往哪改；
AI（Claude）负责前沿检索、数据抓取清洗、写代码、跑回测和消融、搜宿主、做地图和报告。几个转折都来自人的判断：发现模型只比“只看需求”高一点，要求找不吃算力的改进；
发现选点落在空军基地、海上，加了尽调回路。完整 prompt 与对话另附。</p>
</div>

<div>
<figure class="fig2"><img src="data:image/jpeg;base64,{WEB}"><figcaption>图 2　交互网页地图（web/index.html，双击即开）：点击站点弹出配置、并网、经济指标，并可跳转 Google 地图与街景。</figcaption></figure>

<h2><i>4</i>验证与待线下确认</h2>
<p><b>回测</b>：2016–2025 年每年 1 月 1 日只用当时可见的网络和竞品打分，对照当年真实开站；2019–2025 年滚动样本外，每年只用过去数据重新校准。</p>
<table class="mt"><tr><th>滚动样本外 2019–2025</th><th>AUC</th><th>命中率*</th><th>随机命中率</th></tr>
<tr><td>都市圈（学习型筛选）</td><td>0.836</td><td>0.78</td><td>0.28</td></tr>
<tr><td>都市圈（只看需求，对照）</td><td>0.751</td><td>—</td><td>—</td></tr>
<tr><td>高速走廊</td><td>0.673</td><td>0.31</td><td>0.15</td></tr></table>
<p class="note">*前 K 名中，当年或次年 15 km（走廊 25 km）内出现特斯拉新站的比例。</p>
<p><b>前向验证</b>：只用已开通网络打分，都市圈前 50 名有 <b>88%</b> 落在特斯拉当前在建/规划站点 15 km 内（随机 38%）；走廊 23%（随机 10%）。</p>
<p><b>不确定性</b>：NPV 依赖需求与电价假设（没有公开的单站利用率），排序比绝对金额更可信。最需要线下确认的两件事：</p>
<ul>
<li><b>电网余量</b>：没有全国承载能力数据，用的是情景值。压力测试显示余量只有假设的 30% 时，FL、MI、OR 各需加 1 台 Megapack，AZ 要改为等电网升级，需向当地电力公司调承载能力图核实。</li>
<li><b>宿主</b>：温彻斯特 Boulder Crossing 是新建商业中心，要确认是否还有空地块；Walmart 在自建快充，谈判有变数。另外 Kroger 需谈夜间通行，Merchant Row 规模偏小（备选 Greenfield Center）。</li></ul>
</div>
</div>

<h2><i>5</i>加入团队后第一个月</h2>
<div class="grid3" style="grid-template-columns:repeat(5,1fr);margin-top:2px">
<div class="box"><h3>a. 拆解现有玩家</h3>验证特斯拉近年选址逻辑，并扩展到 Electrify America、Ionna、Walmart、Buc-ee's 等，看各自偏好与空白</div>
<div class="box"><h3>b. 积累数据源</h3>电力公司承载能力图、真实利用率 / 交易数据、POI 与地块、交通计数，逐步替换现有假设参数</div>
<div class="box"><h3>c. 大量消融实验</h3>按都市圈 / 走廊 / 气候区 / 电价区分别定制模型与超参数，看哪些因素在哪些场景真正起作用</div>
<div class="box"><h3>d. 面对面交流</h3>和业务、产品同事及客户深聊当前进展、瓶颈与方向，把业务规则和约束校准到真实决策</div>
<div class="box"><h3>e. 沉淀工作流</h3>把数据更新、回测、选点、尽调、出图出报告做成可复用的代码 / skill / harness，新问题能快速复跑</div>
</div>
<p class="note" style="margin-top:8px;border-top:1px solid #e1e6ee;padding-top:5px">交付物：全流程代码（数据获取 → 回测 → 选点 → 尽调 → 出图出报告，<code>python run_all.py</code> 一键复跑）、
<code>final_10_sites.csv</code>（含备选宿主、电网运行策略、定价）、交互地图 <code>web/index.html</code>。经济指标为模型估算，需以实际电价、利用率和电网数据校准。</p>
</div>
</body></html>"""


def main():
    from playwright.sync_api import sync_playwright
    src = OUT / "report.html"
    src.write_text(HTML, encoding="utf-8")
    pdf = OUT / "特斯拉超充选址_提交报告.pdf"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(src.as_uri())
        pg.wait_for_timeout(500)
        pg.pdf(path=str(pdf), format="A4", print_background=True, prefer_css_page_size=True)
        b.close()
    print(pdf)


if __name__ == "__main__":
    main()
