import logging
import json
from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import azure.functions as func

# ---------- 配置区 ----------
LATITUDE = 35.6895
LONGITUDE = 139.6917
TIMEZONE = "Asia/Tokyo"

# 创建 FunctionApp 实例
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# 天气分类函数
def categorize_weather(code: int) -> str:
    """
    将 weathercode 分类为 'sunny', 'cloudy' 或 'rainy'.
    0,1,2 => sunny; 3,45,48 => cloudy; 其他 => rainy
    """
    if code in (0, 1, 2):
        return "sunny"
    elif code in (3, 45, 48):
        return "cloudy"
    else:
        return "rainy"

@app.route(route="GetHistorical", methods=["GET"])
def get_historical(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GetHistorical triggered")
    # 计算当前东京日期，并取两天前
    utc_now = datetime.utcnow()
    tokyo_now = utc_now + timedelta(hours=9)
    hist_date = (tokyo_now.date() - timedelta(days=2)).isoformat()

    # 调用历史天气 API，获取所需字段
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": hist_date,
        "end_date":   hist_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weathercode"
        ]),
        "timezone": TIMEZONE
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        d = resp.json().get("daily", {})
    except requests.exceptions.RequestException as e:
        logging.error(f"调用历史天气 API 出错: {e}")
        return func.HttpResponse("历史天气数据获取失败", status_code=500)

    # 构建 DataFrame
    df = pd.DataFrame({
        "date": d.get("time", []),
        "temperature_2m_mean": d.get("temperature_2m_mean", []),
        "temperature_2m_max": d.get("temperature_2m_max", []),
        "temperature_2m_min": d.get("temperature_2m_min", []),
        "precipitation_sum": d.get("precipitation_sum", []),
        "weathercode": d.get("weathercode", [])
    })
    # 添加天气分类列
    df["weather"] = df["weathercode"].apply(lambda c: categorize_weather(int(c)))

    
    # 返回 JSON
    return func.HttpResponse(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False),
        status_code=200,
        mimetype="application/json"
    )
