import streamlit as st
import pandas as pd
from modules.logger import log_action, get_log, clear_log, export_log
from modules.data_loader import load_data
from modules.preprocessing import (
    fill_missing,
    get_numeric_columns,
    remove_outliers,
    visualize_missing_values
)
from modules.analysis import (
    perform_t_test,
    perform_mannwhitneyu,
    perform_kruskal,
    compute_correlations,
    plot_correlation_heatmap
)
from modules.visualization import plot_histogram
from modules.gbd_module import (
    plot_gbd_trend, plot_location_comparison, plot_disease_composition,
    plot_gbd_map, plot_gbd_animated_line,
    get_top_n_summary, get_growth_summary,
    plot_gbd_by_sex, plot_gbd_by_age
)
import statsmodels.api as sm
import io
import base64

st.set_page_config(page_title="医学数据分析平台", layout="wide")
st.title("医学数据统计分析与可视化平台")

menu = st.sidebar.radio("选择功能模块", [
    "数据上传", "预处理", "异常值检测", "t检验", "统计拓展分析",
    "回归分析", "主成分分析", "聚类分析", "模型评估","生存分析",
    "模型预测", "统计辅助分析", "特征选择", "数据导出",
    "绘图展示", "GBD分析", "报告总览", "生成报告"
])

# 上传数据
# 上传数据
if menu == "数据上传":
    st.subheader("📤 数据上传")
    uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = load_data(uploaded_file)

            # ✅ 风险点 1：数据为空
            if df.empty:
                st.error("❌ 数据为空，请重新上传有效的 CSV 或 Excel 文件。")
                st.stop()

            # ✅ 风险点 2：无数值型字段
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                st.error("❌ 当前数据中不包含任何数值型字段，无法进行后续分析。")
                st.stop()

            st.session_state["df"] = df
            st.success(f"✅ 数据加载成功，共 {df.shape[0]} 行，{df.shape[1]} 列。")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ 文件读取失败，请检查文件格式或内容。\n错误信息: {e}")


# 数据预处理
elif menu == "预处理":
    if "df" in st.session_state:
        df = st.session_state["df"]
        df_cleaned = fill_missing(df)
        st.write("缺失值处理后数据：")
        st.dataframe(df_cleaned)
        st.session_state["df"] = df_cleaned
    else:
        st.warning("请先上传数据")

#异常值检测
elif menu == "异常值检测":
    from modules.preprocessing import (
        get_numeric_columns,
        detect_outliers_summary,
        remove_outliers,
        visualize_missing_values,
        plot_outlier_boxplot
    )

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        selected = st.multiselect("选择字段检测异常值", numeric_cols)

        if selected and st.button("可视化异常值"):
            fig = plot_outlier_boxplot(df, selected)
            st.plotly_chart(fig)

            outlier_idx = detect_outliers_summary(df, selected)
            st.write(f"共检测到 {len(outlier_idx)} 个包含异常值的样本")
            st.dataframe(df.loc[outlier_idx])

        if selected and st.button("删除异常值并更新数据"):
            df_cleaned = remove_outliers(df, selected)
            st.session_state["df"] = df_cleaned
            st.success("异常值已删除，数据已更新")
            st.dataframe(df_cleaned)

        st.subheader("📉 缺失值分布")
        fig = visualize_missing_values(df)
        if fig:
            st.plotly_chart(fig)
        else:
            st.info("无缺失字段")
    else:
        st.warning("请先上传数据")


# 统计检验
elif menu == "t检验":
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

        st.subheader("🎯 t 检验（仅适用于两个分组比较）")
        group_column = st.selectbox("选择分组变量", df.columns, key="t_group_col")
        target_column = st.selectbox("选择数值字段", numeric_columns, key="t_target_col")

        if group_column and target_column:
            group_values = df[group_column].dropna().unique().tolist()

            # 显示所有可选组，用户手动选择两个
            selected_groups = st.multiselect(
                "选择两个具体分组值进行比较",
                group_values,
                default=group_values[:2],
                key="t_selected_groups"
            )

            if len(selected_groups) == 2:
                g1, g2 = selected_groups
                data1 = df[df[group_column] == g1][target_column].dropna()
                data2 = df[df[group_column] == g2][target_column].dropna()

                if st.button("执行 t 检验"):
                    from scipy.stats import ttest_ind
                    stat, p = ttest_ind(data1, data2, equal_var=False)
                    st.success(f"{g1} vs {g2} 的 t 检验结果：")
                    st.write(f"t = {stat:.4f}, p = {p:.4f}")
            elif len(selected_groups) > 2:
                st.warning("⚠️ 请只选择两个分组值")
            elif len(selected_groups) < 2:
                st.info("请从该分组字段中选择两个组进行比较")
    else:
        st.warning("请先上传数据")

#统计拓展分析
elif menu == "统计拓展分析":
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        group_cols = df.columns.tolist()

        st.subheader("非参数检验")
        group_col = st.selectbox("选择分组字段", group_cols, key="nonparam_group")
        numeric_col = st.selectbox("选择数值字段", numeric_cols, key="nonparam_value")
        method = st.selectbox("选择检验方法", ["Mann-Whitney U", "Kruskal-Wallis"])
        if st.button("执行非参数检验"):
            if method == "Mann-Whitney U":
                stat, p = perform_mannwhitneyu(df, numeric_col, group_col)
            else:
                stat, p = perform_kruskal(df, numeric_col, group_col)
            st.success(f"{method} 结果：统计量 = {stat:.4f}，p值 = {p:.4f}")

        st.subheader("相关性分析与热力图")
        corr_method = st.selectbox("相关系数类型", ["pearson", "spearman"])
        if st.button("计算并显示热力图"):
            corr_matrix = compute_correlations(df, method=corr_method)
            fig = plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig)

            # ✅ 保存图像为 base64，用于报告嵌入
            img_buf = io.BytesIO()
            fig.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_heatmap_img"] = img_base64

    else:
        st.warning("请先上传数据")

#回归分析
elif menu == "回归分析":
    from modules.analysis import (
        linear_regression, logistic_regression,
        multivariable_linear_regression, multivariable_logistic_regression
    )

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)

        st.subheader("📈 回归模型设置")
        model_type = st.radio("选择模型类型", ["线性回归", "逻辑回归"])
        x_cols = st.multiselect("选择一个或多个自变量（X）", numeric_cols, key="reg_multi_x")
        y_col = st.selectbox("选择因变量（Y）", numeric_cols, key="reg_multi_y")

        if st.button("训练回归模型"):
            if not x_cols or not y_col:
                st.warning("⚠️ 请确保至少选择一个自变量和一个因变量。")
                st.stop()
            # 字段缺失值检测（防止模型训练失败）
            subset = df[x_cols + [y_col]]
            if subset.isnull().any().any():
                st.error("❌ 所选字段中包含缺失值，请先在预处理模块进行处理。")
                st.stop()
            try:
                if model_type == "线性回归":
                    model = multivariable_linear_regression(df, x_cols, y_col)
                else:
                    model = multivariable_logistic_regression(df, x_cols, y_col)

                # ✅ 保存模型和特征列到会话状态
                st.session_state["trained_model"] = model
                st.session_state["model_features"] = x_cols
                st.success("模型训练成功！")
                log_action(f"训练了一个{model_type}模型，目标变量为：{y_col}，特征包括：{x_cols}")
                st.markdown("### 📋 回归模型摘要")
                st.text(model.summary())

                # === 可视化开始 ===
                import numpy as np
                from modules.visualization import (
                    plot_fitted_vs_actual, plot_residuals, plot_logistic_prediction
                )

                with st.expander("📊 模型可视化"):
                    X = df[x_cols].dropna()
                    y = df[y_col].loc[X.index]
                    X_const = sm.add_constant(X)

                    if model_type == "线性回归":
                        y_pred = model.predict(X_const)
                        st.plotly_chart(plot_fitted_vs_actual(y, y_pred))
                        st.plotly_chart(plot_residuals(y, y_pred))

                    elif model_type == "逻辑回归":
                        y_pred_prob = model.predict(X_const)
                        if X.shape[1] == 1:
                            st.plotly_chart(plot_logistic_prediction(X, y, y_pred_prob))
                        else:
                            st.info("逻辑回归可视化仅支持单个自变量")
                # === 可视化结束 ===


            except Exception as e:
                st.error(f"模型训练失败：{e}")
    else:
        st.warning("请先上传数据")

#主成分分析
elif menu == "主成分分析":
    from modules.pca_module import compute_pca, plot_pca_2d

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        selected_cols = st.multiselect("选择数值字段进行PCA", numeric_cols)

        if len(selected_cols) < 2:
            st.warning("⚠️ 请至少选择两个数值字段进行PCA降维。")
            st.stop()
        elif st.button("执行主成分分析"):
            try:
                pca_df, var_ratio = compute_pca(df, selected_cols)
                st.success(f"前两个主成分累计解释方差为：{var_ratio[:2].sum():.2%}")
                fig = plot_pca_2d(pca_df)
                st.plotly_chart(fig)

                # ✅ 保存图像为 base64，用于报告嵌入
                img_buf = io.BytesIO()
                fig.write_image(img_buf, format="png", engine="kaleido")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                st.session_state["report_pca_img"] = img_base64

            except Exception as e:
                st.error(f"PCA分析失败：{e}")

            except Exception as e:
                st.error(f"PCA分析失败：{e}")
    else:
        st.warning("请先上传数据")

#聚类分析
elif menu == "聚类分析":
    from modules.clustering import perform_kmeans, plot_kmeans_pca

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        selected_cols = st.multiselect("选择聚类变量", numeric_cols)
        k = st.slider("选择聚类数量（K）", 2, 10, 3)

        if len(selected_cols) < 2:
            st.warning("⚠️ 请至少选择两个变量用于聚类分析。")
            st.stop()
        elif st.button("执行聚类分析"):
            pca_df, inertia = perform_kmeans(df, selected_cols, n_clusters=k)
            try:
                pca_df, inertia = perform_kmeans(df, selected_cols, n_clusters=k)
                st.success(f"KMeans 聚类完成，聚类数：{k}，总内聚度（Inertia）：{inertia:.2f}")
                fig = plot_kmeans_pca(pca_df)
                st.plotly_chart(fig)

                # ✅ 保存图像为 base64，用于报告嵌入
                img_buf = io.BytesIO()
                fig.write_image(img_buf, format="png", engine="kaleido")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                st.session_state["report_cluster_img"] = img_base64

            except Exception as e:
                st.error(f"聚类分析失败：{e}")

            except Exception as e:
                st.error(f"聚类分析失败：{e}")
    else:
        st.warning("请先上传数据")

#模型评估
elif menu == "模型评估":
    from modules.evaluation import (
        evaluate_regression, evaluate_classification, plot_confusion_matrix
    )

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        st.subheader("📈 模型评估")

        model_type = st.radio("选择模型类型", ["线性回归", "逻辑回归"])
        x_cols = st.multiselect("选择特征变量（X）", numeric_cols, key="eval_x")
        y_col = st.selectbox("选择目标变量（Y）", numeric_cols, key="eval_y")

        if x_cols and y_col and st.button("评估模型"):
            try:
                from sklearn.linear_model import LinearRegression, LogisticRegression

                X = df[x_cols].dropna()
                y = df[y_col].loc[X.index]

                if model_type == "线性回归":
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    metrics = evaluate_regression(y, y_pred)
                    st.success("评估结果如下：")
                    st.json(metrics)

                elif model_type == "逻辑回归":
                    model = LogisticRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    y_prob = model.predict_proba(X)[:, 1]
                    metrics = evaluate_classification(y, y_pred)
                    st.success("评估结果如下：")
                    st.json(metrics)
                    fig = plot_confusion_matrix(y, y_pred)
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"模型评估失败：{e}")
    else:
        st.warning("请先上传数据")

#生存分析
elif menu == "生存分析":
    from modules import survival_analysis
    survival_analysis.run_survival_analysis()

#模型预测
elif menu == "模型预测":
    from modules.prediction import predict_with_model
    import joblib

    st.subheader("📡 模型预测接口")

    if "df" not in st.session_state:
        st.warning("请先上传训练数据并建立模型")
    else:
        if "trained_model" not in st.session_state or "model_features" not in st.session_state:
            st.warning("未找到已训练模型，请先在回归分析中训练模型")
        else:
            uploaded = st.file_uploader("上传新数据文件（用于预测）", type=["csv", "xlsx"])
            if uploaded:
                new_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                st.dataframe(new_df.head())

                if st.button("执行预测"):
                    try:
                        model = st.session_state["trained_model"]
                        x_cols = st.session_state["model_features"]
                        result_df = predict_with_model(model, new_df, x_cols)
                        st.success("预测完成，结果如下：")
                        st.dataframe(result_df.head())

                        csv = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("📥 下载预测结果 CSV", data=csv, file_name="prediction_result.csv")

                    except Exception as e:
                        st.error(f"预测失败：{e}")

#统计辅助分析
elif menu == "统计辅助分析":
    from modules.analysis import test_normality, plot_qq
    from modules.evaluation import plot_roc_curve

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        st.subheader("📏 正态性检验")
        selected = st.selectbox("选择字段", numeric_cols)
        if st.button("执行Shapiro-Wilk检验"):
            stat, p = test_normality(df, selected)
            st.info(f"Shapiro-Wilk 检验结果: 统计量 = {stat:.4f}, p值 = {p:.4f}")
            fig = plot_qq(df, selected)
            st.plotly_chart(fig)

            # ✅ 保存图像为 base64，用于报告嵌入
            img_buf = io.BytesIO()
            fig.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_qq_img"] = img_base64

        st.markdown("---")
        st.subheader("📉 分类模型 ROC 曲线")
        x_cols = st.multiselect("特征列（X）", numeric_cols, key="roc_x")
        y_col = st.selectbox("目标列（Y，0/1）", numeric_cols, key="roc_y")

        if x_cols and y_col and st.button("绘制ROC曲线"):
            from sklearn.linear_model import LogisticRegression

            X = df[x_cols].dropna()
            y = df[y_col].loc[X.index]
            model = LogisticRegression()
            model.fit(X, y)
            y_prob = model.predict_proba(X)[:, 1]
            fig, auc_val = plot_roc_curve(y, y_prob)
            st.plotly_chart(fig)
            st.success(f"AUC = {auc_val:.4f}")

            # ✅ 保存图像为 base64，用于报告嵌入
            img_buf = io.BytesIO()
            fig.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_roc_img"] = img_base64

    else:
        st.warning("请先上传数据")

#特征选择
elif menu == "特征选择":
    from modules.feature_selection import filter_feature_selection, wrapper_feature_importance

    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = get_numeric_columns(df)
        st.subheader("🧠 自动特征选择")

        x_cols = st.multiselect("选择候选特征字段", numeric_cols)
        y_col = st.selectbox("选择分类目标字段", numeric_cols)

        if x_cols and y_col and st.button("执行特征选择"):
            try:
                result_filter = filter_feature_selection(df, x_cols, y_col)
                result_wrapper = wrapper_feature_importance(df, x_cols, y_col)

                st.markdown("### 📊 基于F检验的特征筛选（Filter法）")
                st.dataframe(result_filter)

                st.markdown("### 🧠 基于模型的重要性评分（Wrapper法）")
                st.dataframe(result_wrapper)

            except Exception as e:
                st.error(f"特征选择失败：{e}")
    else:
        st.warning("请先上传数据")

#数据导出
elif menu == "数据导出":
    st.subheader("📤 数据导出功能")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.dataframe(df.head())

        filename = st.text_input("输入导出文件名（无需加后缀）", value="processed_data")
        if st.button("导出为 CSV 文件"):
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 点击下载",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
    else:
        st.warning("请先上传并处理数据")

# 绘图模块
elif menu == "绘图展示":
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("📊 单变量直方图")
        column = st.selectbox("选择字段绘制直方图", df.select_dtypes("number").columns, key="hist_col")
        fig = plot_histogram(df, column)
        st.plotly_chart(fig)

        # 保存为 base64
        img_buf = io.BytesIO()
        fig.write_image(img_buf, format="png", engine="kaleido")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        st.session_state["report_hist_img"] = img_base64

        st.subheader("📌 多变量雷达图")

        all_cols = df.columns.tolist()
        group_col = st.selectbox("选择分组变量", all_cols, key="radar_group")
        value_cols = st.multiselect("选择数值字段（3个以上）", df.select_dtypes("number").columns.tolist(),
                                    key="radar_values")

        from modules.visualization import plot_radar_chart, export_plotly_figure_to_png, export_dataframe_to_csv_bytes

        if not group_col:
            st.warning("⚠️ 请先选择一个分组字段。")
        elif len(value_cols) < 3:
            st.warning("⚠️ 请至少选择 3 个数值字段再绘制雷达图。")
        else:
            fig_radar = plot_radar_chart(df, group_col, value_cols)

            # 👉 图像保存为 base64，用于报告嵌入
            img_buf = io.BytesIO()
            fig_radar.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_radar_img"] = img_base64

            st.plotly_chart(fig_radar)
            log_action(f"绘制了雷达图，分组字段为：{group_col}，分析指标：{value_cols}")

            radar_data = df[[group_col] + value_cols].dropna()

            # 导出按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 导出雷达图为 PNG"):
                    img_bytes = export_plotly_figure_to_png(fig_radar)
                    st.download_button("点击下载雷达图", data=img_bytes, file_name="radar_chart.png", mime="image/png")

            with col2:
                csv_bytes = export_dataframe_to_csv_bytes(radar_data)
                st.download_button("📄 导出数据为 CSV", data=csv_bytes, file_name="radar_data.csv", mime="text/csv")

    else:
        st.warning("请先上传数据")


# GBD 分析
elif menu == "GBD分析":
    if "df" in st.session_state:
        df = st.session_state["df"]
        # GBD字段结构完整性校验
        required_cols = {"cause_name", "location_name", "year", "val", "metric_name"}
        if not required_cols.issubset(df.columns):
            st.error("❌ 当前数据缺失 GBD 所需字段，无法进行分析。")
            st.stop()
        if not all(col in df.columns for col in ["cause_name", "location_name", "year", "val", "metric_name"]):
            st.error("当前数据不是标准 GBD 格式，请检查字段是否包括：cause_name, location_name, year, val, metric_name")
        else:
            st.subheader("📈 疾病趋势图")
            cause = st.selectbox("选择疾病", df["cause_name"].unique())
            location = st.selectbox("选择地区", df["location_name"].unique())
            fig1 = plot_gbd_trend(df, cause, location)
            st.plotly_chart(fig1)

            # 保存为 base64
            img_buf = io.BytesIO()
            fig1.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_gbd_trend_img"] = img_base64

            st.subheader("📊 地区对比图")
            year = st.selectbox("选择年份", sorted(df["year"].unique()))
            fig2 = plot_location_comparison(df, cause, year)
            st.plotly_chart(fig2)

            # 保存为 base64
            img_buf = io.BytesIO()
            fig2.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_gbd_compare_img"] = img_base64

            st.subheader("📊 疾病构成图")
            fig3 = plot_disease_composition(df, location, year)
            st.plotly_chart(fig3)

            # 保存为 base64
            img_buf = io.BytesIO()
            fig3.write_image(img_buf, format="png", engine="kaleido")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            st.session_state["report_gbd_compose_img"] = img_base64

            st.subheader("🗺️ GBD 地图热图")
            metric_map = st.selectbox("选择指标（用于地图显示）", df["metric_name"].unique(), key="gbd_map_metric")
            year_map = st.selectbox("选择年份", sorted(df["year"].unique()), key="gbd_map_year")
            if st.button("显示全球分布图"):
                fig_map = plot_gbd_map(df, year_map, metric_map)
                st.plotly_chart(fig_map)

                # 保存为 base64
                img_buf = io.BytesIO()
                fig_map.write_image(img_buf, format="png", engine="kaleido")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                st.session_state["report_gbd_map_img"] = img_base64

            st.subheader("🎞️ GBD 年度趋势动画图")
            cause_anim = st.selectbox("选择疾病", df["cause_name"].unique(), key="gbd_anim_cause")
            metric_anim = st.selectbox("选择指标", df["metric_name"].unique(), key="gbd_anim_metric")
            if st.button("显示趋势动画图"):
                fig_anim = plot_gbd_animated_line(df, cause_anim, metric_anim)
                st.plotly_chart(fig_anim)

            st.subheader("🧠 智能摘要卡片")

            metric_summary = st.selectbox("选择指标", df["metric_name"].unique(), key="gbd_sum_metric")
            year_summary = st.selectbox("选择年份", sorted(df["year"].unique()), key="gbd_sum_year")
            topn = st.slider("查看负担Top N", 1, 10, 5)

            if st.button("生成Top N地区/疾病摘要"):
                summary_lines = get_top_n_summary(df, metric_summary, year_summary, topn)
                st.markdown("#### 🏆 Top地区-疾病负担")
                for line in summary_lines:
                    st.markdown(f"- {line}")

            st.subheader("👨‍⚕️ GBD 性别分层趋势图")
            cause_sex = st.selectbox("选择疾病（性别分析）", df["cause_name"].unique(), key="gbd_sex_cause")
            metric_sex = st.selectbox("选择指标", df["metric_name"].unique(), key="gbd_sex_metric")
            location_sex = st.selectbox("选择地区", df["location_name"].unique(), key="gbd_sex_loc")
            if st.button("生成按性别趋势图"):
                fig_sex = plot_gbd_by_sex(df, cause_sex, metric_sex, location_sex)
                st.plotly_chart(fig_sex)

                # 保存为 base64
                img_buf = io.BytesIO()
                fig_sex.write_image(img_buf, format="png", engine="kaleido")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                st.session_state["report_gbd_sex_img"] = img_base64

            st.subheader("👶 GBD 年龄段分层趋势图")
            cause_age = st.selectbox("选择疾病（年龄分析）", df["cause_name"].unique(), key="gbd_age_cause")
            metric_age = st.selectbox("选择指标", df["metric_name"].unique(), key="gbd_age_metric")
            location_age = st.selectbox("选择地区", df["location_name"].unique(), key="gbd_age_loc")
            if st.button("生成按年龄趋势图"):
                fig_age = plot_gbd_by_age(df, cause_age, metric_age, location_age)
                st.plotly_chart(fig_age)

                # 保存为 base64
                img_buf = io.BytesIO()
                fig_age.write_image(img_buf, format="png", engine="kaleido")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                st.session_state["report_gbd_age_img"] = img_base64

            st.markdown("---")
            st.markdown("#### 📈 疾病增长变化解读")

            cause_growth = st.selectbox("选择疾病", df["cause_name"].unique(), key="gbd_sum_cause")
            location_growth = st.selectbox("选择地区", df["location_name"].unique(), key="gbd_sum_loc")

            if st.button("生成同比趋势解释"):
                growth_text = get_growth_summary(df, metric_summary, cause_growth, location_growth)
                st.success(growth_text)


    else:
        st.warning("请先上传 GBD 数据")

#报告总览
elif menu == "报告总览":
    from modules.summary_dashboard import create_sample_summary_card, generate_interpretation_text
    from modules.visualization import plot_radar_chart

    st.subheader("📊 图文整合报告页")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # 示例内容：雷达图 + 解读
        group_col = st.selectbox("选择分组变量", df.columns, key="sum_group")
        value_cols = st.multiselect("选择数值字段（3个以上）", df.select_dtypes("number").columns.tolist(), key="sum_val")

        if group_col and len(value_cols) >= 3 and st.button("生成图文卡片"):
            fig = plot_radar_chart(df, group_col, value_cols)
            interp = "\n".join([
                generate_interpretation_text(df, col, group_col) for col in value_cols
            ])
            card = create_sample_summary_card("多变量指标雷达图", interp, fig)

            st.markdown(f"### 🧾 {card['title']}")
            st.plotly_chart(card["figure"])
            st.success(card["text"])
    else:
        st.warning("请先上传数据")

# 项目管理
elif menu == "项目管理":
    from modules.project_state import save_project_state, load_project_state
    st.markdown("---")
    st.markdown("### 📋 当前分析日志记录")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 清空日志"):
            clear_log()
            st.success("日志已清空")

    with col2:
        log_text = export_log()
        if log_text:
            st.download_button("📥 下载日志记录", data=log_text, file_name="log.txt", mime="text/plain")

    logs = get_log()
    if logs:
        for line in logs:
            st.markdown(f"- {line}")
    else:
        st.info("暂无日志记录")

    st.subheader("📦 项目保存与加载")

    # 保存项目
    st.markdown("### 🔐 保存当前分析项目")
    save_name = st.text_input("保存文件名（如：my_analysis.json）", value="my_project.json")
    if st.button("📥 保存项目"):
        try:
            save_project_state(save_name, st.session_state)
            log_action(f"保存了项目文件：{save_name}")
            with open(save_name, "rb") as f:
                st.download_button("点击下载保存文件", f, file_name=save_name, mime="application/json")
        except Exception as e:
            st.error(f"保存失败：{e}")

    st.markdown("---")

    # 加载项目
    st.markdown("### 📂 加载之前保存的项目")
    uploaded_json = st.file_uploader("上传项目文件（.json）", type=["json"])
    if uploaded_json:
        try:
            import tempfile
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
            with open(tmp_path, "wb") as f:
                f.write(uploaded_json.read())
            restored = load_project_state(tmp_path)

            for key, value in restored.items():
                st.session_state[key] = value

            st.success("项目加载成功，当前状态已恢复！")
            st.experimental_rerun()

        except Exception as e:
            st.error(f"加载失败：{e}")

#生成报告
elif menu == "生成报告":
    from modules.report_generator import generate_markdown_report, render_html_report, generate_pdf_from_html

    st.subheader("📝 自动生成分析报告")

    # 示例动态内容构建
    gbd_summary_card = ""

    if "df" in st.session_state:
        df = st.session_state["df"]
        import random
        from modules.gbd_module import get_top_n_summary, get_growth_summary

        try:
            metric = df["metric_name"].unique()[0]
            year = df["year"].max()
            top_lines = get_top_n_summary(df, metric, year, 3)
            top_block = "\n".join([f"- {line}" for line in top_lines])

            cause = df["cause_name"].unique()[0]
            loc = df["location_name"].unique()[0]
            trend_text = get_growth_summary(df, metric, cause, loc)

            gbd_summary_card = f"""
当前分析中，{year} 年 {metric} 指标负担排名前列的地区和疾病包括：
{top_block}

此外，{cause} 在 {loc} 地区的 {metric} 指标 {trend_text}。
            """
        except:
            gbd_summary_card = "GBD 数据不足，无法生成自动结论。"
    default_context = {
        "summary": f"共上传数据 {df.shape[0]} 行，{df.shape[1]} 列。" if "df" in st.session_state else "",
        "test_result": "t 检验与非参数检验结果已完成。",
        "correlation_info": "相关性热图已展示。",
        "gbd_info": "GBD 趋势图、地区比较图、疾病构成图均已可视化。",
        "gbd_summary_card": gbd_summary_card,
        "radar_img": st.session_state.get("report_radar_img"),
        "heatmap_img": st.session_state.get("report_heatmap_img"),
        "cluster_img": st.session_state.get("report_cluster_img"),
        "pca_img": st.session_state.get("report_pca_img"),
        "qq_img": st.session_state.get("report_qq_img"),
        "roc_img": st.session_state.get("report_roc_img"),
        "gbd_trend_img": st.session_state.get("report_gbd_trend_img"),
        "gbd_compare_img": st.session_state.get("report_gbd_compare_img"),
        "gbd_compose_img": st.session_state.get("report_gbd_compose_img"),
        "gbd_map_img": st.session_state.get("report_gbd_map_img"),
        "hist_img": st.session_state.get("report_hist_img"),
        "gbd_sex_img": st.session_state.get("report_gbd_sex_img"),
        "gbd_age_img": st.session_state.get("report_gbd_age_img"),
    }

    md_report = generate_markdown_report(default_context)
    html_report = render_html_report(md_report)

    st.download_button("📥 下载报告（HTML格式）", data=html_report, file_name="analysis_report.html", mime="text/html")

    if st.button("📄 生成 PDF"):
        pdf_bytes = generate_pdf_from_html(html_report)
        log_action("生成并导出了完整分析报告")
        st.download_button("📥 下载报告（PDF格式）", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown(md_report)
