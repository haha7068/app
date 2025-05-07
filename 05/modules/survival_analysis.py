import streamlit as st
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt


def run_survival_analysis():
    st.subheader("📊 生存分析模块")

    if "df" not in st.session_state:
        st.warning("请先上传数据")
        return

    df = st.session_state["df"]

    st.markdown("请选择用于生存分析的字段：")
    time_col = st.selectbox("选择生存时间列", df.columns, key="surv_time")
    event_col = st.selectbox("选择事件状态列（1=事件发生，0=删失）", df.columns, key="surv_event")

    method = st.radio("选择分析方法", ["Kaplan-Meier 生存曲线", "Cox 比例风险模型"])

    if method == "Kaplan-Meier 生存曲线":
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(durations=df[time_col], event_observed=df[event_col])

            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier 生存曲线")
            ax.set_xlabel("时间")
            ax.set_ylabel("生存概率")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"生存曲线绘制失败：{e}")

    elif method == "Cox 比例风险模型":
        covariates = st.multiselect("选择协变量字段", [col for col in df.columns if col not in [time_col, event_col]])

        if covariates and st.button("执行 Cox 回归"):
            try:
                cph_df = df[[time_col, event_col] + covariates].dropna()
                cph = CoxPHFitter()
                cph.fit(cph_df, duration_col=time_col, event_col=event_col)

                st.subheader("📋 Cox 模型摘要")
                st.dataframe(cph.summary)

                fig, ax = plt.subplots()
                cph.plot(ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Cox 模型拟合失败：{e}")
