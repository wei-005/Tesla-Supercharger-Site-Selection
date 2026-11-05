"""Step 7: final deliverable table - addresses, coordinates, charger configuration and operating mode."""
import json
import numpy as np
import pandas as pd
from config import OUT, STALL_KW_DESIGN, COINCIDENCE, MEGAPACK_UNIT_KWH

OPT_CN = {"firm": "常规接入（等待电网升级）", "static_flex": "静态灵活接入（固定功率上限）",
          "dynamic_flex": "动态灵活接入（逐时功率包络）", "bridge": "过渡方案（先动态、后常规）"}
TYPE_CN = {"fuel_travel_center": "加油站/旅行中心", "mall_shopping_center": "购物中心",
           "grocery_bigbox": "大型超市", "hotel_casino": "酒店", "restaurant": "餐饮"}


def ops(r):
    trailer = 4 if r.segment == "corridor" else (2 if r.host_type == "fuel_travel_center" else 0)
    hours24 = "24h" in r.host   # only hosts verified as open 24 h (Google Places hours)
    if r.interconnect == "dynamic_flex":
        grid = (f"申请 {r.request_kw:,.0f} kW，按电力公司逐时包络运行（保底 40%）；限电时柜内功率分配优先给低电量车辆，"
                f"单桩降功率而非关桩")
    elif r.interconnect == "static_flex":
        grid = f"按当前馈线余量签固定上限，站内功率管理将峰值压在上限内；需求增长后再申请扩容"
    elif r.interconnect == "bridge":
        grid = "先以动态包络通电，电网升级完成后转常规接入"
    else:
        grid = "常规接入"
    price = ("分时电价（午间、夜间低价引导错峰），排队超过 80% 占用启用拥堵费，满电后收取占位费；"
             "对所有 NACS 车辆开放，非特斯拉车辆会员/非会员差异定价")
    if r.segment == "corridor":
        price += "；节假日高峰启用排队预约与 80% SOC 上限"
    return trailer, ("24 小时开放" if hours24 else "随宿主营业时间（建议协商 24 小时通行）"), grid, price


def main():
    m = pd.read_csv(OUT / "selected_10_sites_with_hosts.csv").sort_values("npv", ascending=False).reset_index(drop=True)
    rows = []
    for i, r in m.iterrows():
        trailer, hours, grid, price = ops(r)
        rows.append({
            "排名": i + 1, "类型": "城市/都市圈" if r.segment == "metro" else "高速走廊",
            "州": r.state, "县": r.county, "宿主": r.host, "地址": r.address,
            "纬度": round(r.host_lat, 6), "经度": round(r.host_lng, 6), "H3_res6": r.cell6,
            "宿主类型": TYPE_CN.get(r.host_type, r.host_type),
            "超充桩(V4)": int(r.stalls), "其中拖挂直通车位": trailer, "慢充(目的地L2)": int(r.l2),
            "并网方式": OPT_CN[r.interconnect], "申请功率_kW": round(r.request_kw),
            "Megapack台数": int(r.megapacks), "预计通电_月": int(r.months_to_energise),
            "运营时间": hours, "电网运行策略": grid, "定价与运营": price,
            "首年服务电量_MWh": round(r.served_kwh_y1 / 1000), "其中走廊需求_MWh": round(r.corridor_kwh_y1 / 1000),
            "利用率": round(r.utilization, 2), "服务水平": round(r.service_level, 3),
            "CAPEX_万美元": round(r.capex / 1e4), "NPV_万美元": round(r.npv / 1e4),
            "竞品需求占比": round(r.competitor_share_catchment, 2),
            "距最近特斯拉站_km(含在建/规划)": round(r.host_to_nearest_tesla_km, 1),
            "宿主距网格中心_km": round(r.host_to_cell_centre_km, 1), "备选宿主": r.alternate_host,
        })
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "final_10_sites.csv", index=False, encoding="utf-8-sig")
    print(T[["排名", "类型", "州", "宿主", "纬度", "经度", "超充桩(V4)", "慢充(目的地L2)", "并网方式",
             "Megapack台数", "首年服务电量_MWh", "利用率", "NPV_万美元"]].to_string())
    return T


if __name__ == "__main__":
    main()
