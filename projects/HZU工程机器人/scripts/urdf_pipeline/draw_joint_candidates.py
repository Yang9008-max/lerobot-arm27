"""Draw a numbered motion schematic for reviewing measured axis candidates."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from inventory import ROOT


def main():
    data = json.loads((ROOT / "urdf_work/config/joints.json").read_text(encoding="utf-8"))
    image = Image.new("RGB", (1600, 1040), "#f3f6fa")
    d = ImageDraw.Draw(image)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    fonts = {n: ImageFont.truetype(font_path, n) for n in [20, 23, 25, 29, 34, 42]}
    ink, muted, blue = "#14263d", "#53677d", "#2879cb"

    def text(xy, value, size=25, fill=ink):
        d.text(xy, value, font=fonts[size], fill=fill)

    def arrow(start, end, color=blue, width=6):
        import math
        d.line([start, end], fill=color, width=width)
        theta = math.atan2(end[1]-start[1], end[0]-start[0])
        left = (end[0]-18*math.cos(theta-.45), end[1]-18*math.sin(theta-.45))
        right = (end[0]-18*math.cos(theta+.45), end[1]-18*math.sin(theta+.45))
        d.polygon([end, left, right], fill=color)

    text((55, 30), "机械臂关节候选 · 按底座 → 吸盘编号", 42)
    text((57, 98), "依据现有 CAD 测量；动作示意不按比例。编号是本次拟定编号。", 23, muted)
    d.rounded_rectangle((45, 155, 1555, 510), radius=22, fill="white")
    points = [(190, 422), (190, 290), (620, 290), (988, 290), (1260, 290), (1380, 290)]
    d.rounded_rectangle((103, 430, 280, 455), radius=7, fill="#7f91a5")
    d.line([points[0], points[1]], fill="#a8b6c5", width=30)
    d.line([points[1], points[2]], fill="#e3a540", width=31)
    d.line([points[2], points[3]], fill="#4baf90", width=26)
    d.line([points[3], points[4]], fill="#588cca", width=22)
    d.line([points[4], (1470, 290)], fill="#a779c4", width=17)
    d.ellipse((1450, 255, 1490, 325), fill="#a779c4", outline=ink, width=3)
    # Axes perpendicular to this side-view are circles with a center dot.
    for index in [1, 2, 4]:
        x, y = points[index]
        d.ellipse((x-23, y-23, x+23, y+23), fill="white", outline=blue, width=5)
        d.ellipse((x-5, y-5, x+5, y+5), fill=blue)
    arrow((190, 449), (190, 368))
    arrow((955, 290), (1052, 290))
    arrow((1340, 290), (1435, 290))
    label_xy = [(95, 360), (95, 208), (560, 208), (910, 358), (1150, 208), (1330, 358)]
    for n, (x, y) in enumerate(label_xy, 1):
        d.rounded_rectangle((x, y, x+65, y+43), radius=11, fill=blue)
        text((x+12, y+3), f"J{n}", 25, "white")
    text((310, 328), "大臂轴间距：300 mm", 23, muted)
    text((735, 328), "小臂", 23, muted)
    text((1060, 328), "腕部框架", 23, muted)
    text((1395, 427), "吸盘", 23, muted)
    text((75, 470), "⊙ 横向俯仰轴       箭头表示轴线方向，仅用于辨认位置；尚未定义电机正转。", 20, muted)
    d.rounded_rectangle((45, 535, 1555, 930), radius=20, fill="white")
    columns = [75, 170, 480, 830]
    for x, value in zip(columns, ["编号", "候选运动", "驱动对应", "确认依据 / 剩余问题"]):
        text((x, 553), value, 23, muted)
    rows = [
        ("J1", "底座左右回转", "4310 + 25/56 带轮", "底板、回转支撑与轴承垫板同轴"),
        ("J2", "大臂抬起 / 放下", "底座 8009 + 28/56 带轮", "输出轴已定位，不能用电机轴替代"),
        ("J3", "小臂相对大臂屈伸", "肘部 8009", "大小臂连接孔及轴承同轴"),
        ("J4", "整个腕部绕前臂长轴旋转", "GM6020", "固定板与输出连接板同轴"),
        ("J5", "吸盘支架俯仰", "两台 2006 联合驱动候选", "左右支撑孔同轴；差动映射待确认"),
        ("J6", "吸盘绕自身轴旋转", "两台 2006 联合驱动候选", "与 J5 轴相交且垂直；耦合符号待确认"),
    ]
    for i, row in enumerate(rows):
        y = 602 + 51*i
        if i % 2 == 0:
            d.rounded_rectangle((62, y-3, 1538, y+44), radius=7, fill="#eef4fb")
        for x, value in zip(columns, row):
            text((x, y), value, 20)
    text((58, 949), "17 项几何检查通过 ≠ 6 轴控制已标定。两台 2006 的运动组合、全部零位和限位仍待确认。", 23, muted)
    text((58, 990), "真空吸附单独作为开关动作；金矿石属于被抓取物，不计入机器人关节。", 23, muted)
    out = ROOT / "urdf_work/inspection/joint_candidates_overview.png"
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
