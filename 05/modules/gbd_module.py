import plotly.express as px

def plot_gbd_trend(df, cause, location):
    filtered = df[(df['cause_name'] == cause) & (df['location_name'] == location)]
    fig = px.line(
        filtered,
        x="year",
        y="val",
        color="metric_name",
        title=f"{location} - {cause} 指标趋势图"
    )
    return fig
import plotly.express as px

def plot_gbd_trend(df, cause, location):
    filtered = df[(df['cause_name'] == cause) & (df['location_name'] == location)]
    fig = px.line(
        filtered,
        x="year", y="val", color="metric_name",
        title=f"{location} - {cause} 指标趋势图"
    )
    return fig

def plot_location_comparison(df, cause, year):
    filtered = df[(df['cause_name'] == cause) & (df['year'] == year)]
    fig = px.bar(
        filtered,
        x="location_name", y="val", color="metric_name",
        title=f"{year}年 - {cause} 各地区比较",
        labels={"val": "负担值"}
    )
    return fig

def plot_disease_composition(df, location, year):
    filtered = df[(df['location_name'] == location) & (df['year'] == year)]
    fig = px.bar(
        filtered,
        x="cause_name", y="val", color="metric_name",
        title=f"{location} - {year}年 疾病构成",
        labels={"val": "负担值"}
    )
    return fig
import plotly.express as px

def plot_gbd_map(df, year, metric):
    df_map = df[(df["year"] == year) & (df["metric_name"] == metric)]
    fig = px.choropleth(
        df_map,
        locations="location_name",
        locationmode="country names",
        color="val",
        hover_name="location_name",
        color_continuous_scale="Reds",
        title=f"{year} 年 - {metric} 指标的全球分布图",
        labels={"val": metric}
    )
    return fig

def plot_gbd_animated_line(df, cause, metric):
    df_anim = df[df["cause_name"] == cause]
    fig = px.line(
        df_anim,
        x="year", y="val",
        color="location_name",
        animation_frame="year",
        title=f"{cause} - {metric} 随时间演化（按地区）",
        labels={"val": metric}
    )
    return fig
def get_top_n_summary(df, metric, year, top_n=5):
    filtered = df[(df["year"] == year) & (df["metric_name"] == metric)]
    top = filtered.sort_values("val", ascending=False).head(top_n)
    result = [
        f"{i+1}. {row['location_name']} - {row['cause_name']}（{row['val']:.2f}）"
        for i, row in top.iterrows()
    ]
    return result

def get_growth_summary(df, metric, cause, location):
    data = df[(df["metric_name"] == metric) & (df["cause_name"] == cause) & (df["location_name"] == location)]
    if data["year"].nunique() < 2:
        return "数据不足以计算增长趋势"
    data_sorted = data.sort_values("year")
    latest = data_sorted.iloc[-1]
    previous = data_sorted.iloc[-2]
    change = latest["val"] - previous["val"]
    pct = (change / previous["val"]) * 100 if previous["val"] != 0 else 0
    direction = "上升" if change > 0 else "下降"
    return f"{latest['year']}年相较于{previous['year']}年，{location} 的 {cause} {metric} {direction}了 {abs(pct):.2f}%"
import plotly.express as px

def plot_gbd_by_sex(df, cause, metric, location):
    df_filtered = df[
        (df["cause_name"] == cause) &
        (df["metric_name"] == metric) &
        (df["location_name"] == location)
    ]
    fig = px.line(
        df_filtered,
        x="year", y="val", color="sex_name",
        markers=True,
        title=f"{location} - {cause} ({metric}) 按性别趋势图",
        labels={"val": metric, "sex_name": "性别"}
    )
    return fig

def plot_gbd_by_age(df, cause, metric, location):
    df_filtered = df[
        (df["cause_name"] == cause) &
        (df["metric_name"] == metric) &
        (df["location_name"] == location)
    ]
    fig = px.line(
        df_filtered,
        x="year", y="val", color="age_name",
        markers=True,
        title=f"{location} - {cause} ({metric}) 按年龄段趋势图",
        labels={"val": metric, "age_name": "年龄段"}
    )
    return fig
