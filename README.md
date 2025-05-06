# 医学数据统计分析与可视化平台

本项目是一个基于 Python 和 Streamlit 开发的交互式医学数据分析与可视化平台，旨在为医学研究者、公共卫生从业者、科研人员提供一个集数据清洗、统计分析、机器学习建模、GBD 数据分析和图表展示于一体的综合平台。

---

## 🔧 功能模块

* **数据上传与预处理**：支持 CSV、Excel 文件，缺失值与异常值处理
* **统计分析**：t 检验、ANOVA、非参数检验、相关性分析等
* **机器学习建模**：线性/逻辑回归、XGBoost、特征选择与模型评估
* **主成分分析与聚类分析**：支持 PCA 降维、KMeans 聚类等可视化
* **GBD 数据分析**：多维度疾病负担趋势、热力图、TopN 排名、动画趋势
* **自动报告生成**：Markdown + HTML + PDF，一键生成图文并茂的医学分析报告
* **项目管理功能**：操作日志记录、项目状态保存与恢复

---

## 🧱 项目结构

```
├── app.py                  # 主程序入口
├── requirements.txt        # 依赖包说明
├── modules/                # 各功能模块封装
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── gbd_module.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── feature_selection.py
│   ├── report_generator.py
│   ├── summary_dashboard.py
│   └── logger.py
```

---

## 📦 安装依赖

请使用 Python 3.9+，建议创建虚拟环境后运行：

```bash
pip install -r requirements.txt
```

如需生成 PDF 报告，请确保安装并配置 `wkhtmltopdf` 工具。

---

## ▶️ 启动平台

```bash
streamlit run app.py
```

默认将在浏览器中打开：`http://localhost:8501`

---

## 📚 使用到的主要第三方库

| 库名称                  | 用途说明                     |
| -------------------- | ------------------------ |
| streamlit            | Web 应用前端交互               |
| pandas / numpy       | 数据处理                     |
| scipy / statsmodels  | t检验、ANOVA、回归建模           |
| lifelines            | 生存分析（KM曲线、Cox回归）         |
| sklearn / xgboost    | 机器学习建模与评估                |
| plotly               | 交互式图表绘制                  |
| seaborn / matplotlib | 热力图与静态图辅助                |
| markdown2 / jinja2   | 报告文本与 HTML 模板引擎          |
| pdfkit               | PDF 报告生成（配合 wkhtmltopdf） |

---

## 🚀 云端部署建议

支持部署至：

* [Streamlit Community Cloud](https://share.streamlit.io)
* [Render.com](https://render.com)
* 自建服务器（需配置 Python + Streamlit）

---

## 📄 License

MIT License

---

## 🙋 联系作者

如需学术交流或协助部署，请联系：`235890532qq.com`

---

> 本项目为《基于 Python 的医学数据统计分析与可视化平台构建》课题实践成果，适合用于医学科研数据分析、教学演示及科学研究支撑。
