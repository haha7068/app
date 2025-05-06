from jinja2 import Template
import datetime
import markdown2

def generate_markdown_report(context):
    md = f"# 医学数据分析报告\n"
    md += f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    md += f"## 数据摘要\n{context.get('summary', '')}\n\n"

    if "test_result" in context:
        md += "## 统计检验结果\n" + context["test_result"] + "\n"

    if "correlation_info" in context:
        md += "## 相关性分析结果\n" + context["correlation_info"] + "\n"

    if "radar_img" in context:
        md += "## 雷达图\n"
        md += f"![雷达图](data:image/png;base64,{context['radar_img']})\n"

    if "heatmap_img" in context:
        md += "## 相关性热图\n"
        md += f"![热力图](data:image/png;base64,{context['heatmap_img']})\n"

    if "cluster_img" in context:
        md += "## 聚类分析图\n"
        md += f"![聚类图](data:image/png;base64,{context['cluster_img']})\n"

    if "pca_img" in context:
        md += "## 主成分分析图（PCA）\n"
        md += f"![PCA图](data:image/png;base64,{context['pca_img']})\n"

    if "qq_img" in context:
        md += "## 正态性检验 QQ 图\n"
        md += f"![QQ图](data:image/png;base64,{context['qq_img']})\n"

    if "roc_img" in context:
        md += "## 分类模型 ROC 曲线图\n"
        md += f"![ROC曲线](data:image/png;base64,{context['roc_img']})\n"

    if "gbd_info" in context:
        md += "## GBD 数据分析\n" + context["gbd_info"] + "\n"
    if "hist_img" in context:
        md += "## 单变量直方图\n"
        md += f"![直方图](data:image/png;base64,{context['hist_img']})\n"

    if "gbd_sex_img" in context:
        md += "## 👩‍⚕️ GBD 性别分层趋势图\n"
        md += f"![性别图](data:image/png;base64,{context['gbd_sex_img']})\n"

    if "gbd_age_img" in context:
        md += "## 👶 GBD 年龄分层趋势图\n"
        md += f"![年龄图](data:image/png;base64,{context['gbd_age_img']})\n"

    if "gbd_trend_img" in context:
        md += "## 📈 GBD 疾病趋势图\n"
        md += f"![趋势图](data:image/png;base64,{context['gbd_trend_img']})\n"

    if "gbd_compare_img" in context:
        md += "## 📊 GBD 地区对比图\n"
        md += f"![地区对比图](data:image/png;base64,{context['gbd_compare_img']})\n"

    if "gbd_compose_img" in context:
        md += "## 📊 GBD 疾病构成图\n"
        md += f"![疾病构成图](data:image/png;base64,{context['gbd_compose_img']})\n"

    if "gbd_map_img" in context:
        md += "## 🗺️ GBD 地图热图\n"
        md += f"![地图图](data:image/png;base64,{context['gbd_map_img']})\n"

    if "gbd_summary_card" in context:
        md += "## GBD 智能解读段落\n"
        md += context["gbd_summary_card"] + "\n"

    return md

def render_html_report(markdown_text):
    html_body = markdown2.markdown(markdown_text)
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>医学分析报告</title>
        <style>
            body { font-family: "Helvetica", sans-serif; padding: 20px; }
            h1, h2 { color: #2c3e50; }
        </style>
    </head>
    <body>
    {{ content }}
    </body>
    </html>
    """
    return Template(template).render(content=html_body)
import pdfkit
import os
import tempfile

def generate_pdf_from_html(html_content):
    config = None
    if os.name == "nt":
        path = r"D:\wkhtmltopdf\bin\wkhtmltopdf.exe"  # 请根据实际路径修改
        if os.path.exists(path):
            config = pdfkit.configuration(wkhtmltopdf=path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        temp_pdf_path = tmpfile.name  # 仅保留路径，不保持文件句柄

    try:
        pdfkit.from_string(html_content, temp_pdf_path, configuration=config)
        with open(temp_pdf_path, "rb") as f:
            pdf_data = f.read()
    finally:
        os.remove(temp_pdf_path)  # 确保不留临时文件

    return pdf_data
