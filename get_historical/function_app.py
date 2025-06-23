import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import requests
import azure.functions as func

# ---------- 配置区 ----------
LATITUDE = 35.6895
LONGITUDE = 139.6917
TIMEZONE = "Asia/Tokyo"

# 天气分类函数
def categorize_weather(code: int) -> str:
    if code in (0, 1, 2):
        return "sunny"
    elif code in (3, 45, 48):
        return "cloudy"
    else:
        return "rainy"

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="GetHistorical", methods=["GET", "POST"])
def get_historical(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GetHistorical triggered")

    # 1. 读取用户传入的日期参数
    date_str = req.params.get("date")
    if not date_str:
        try:
            body = req.get_json()
            date_str = body.get("date")
        except ValueError:
            pass

    if not date_str:
        return func.HttpResponse(
            "请提供参数 date（格式 YYYY-MM-DD），例如 GET /api/GetHistorical?date=2025-06-20",
            status_code=400
        )

    # 2. 解析并计算目标查询日期（前两天）
    try:
        user_date = datetime.fromisoformat(date_str).date()
    except ValueError:
        return func.HttpResponse(
            "日期格式不正确，请使用 ISO 格式 YYYY-MM-DD",
            status_code=400
        )
    hist_date = (user_date - timedelta(days=2)).isoformat()

    # 3. 调用历史天气 API
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
        daily = resp.json().get("daily", {})
    except requests.exceptions.RequestException as e:
        logging.error(f"历史天气 API 调用失败: {e}")
        return func.HttpResponse("历史天气数据获取失败", status_code=500)

    # 4. 构建并返回结果
    df = pd.DataFrame({
        "date":                 daily.get("time", []),
        "temperature_2m_mean":  daily.get("temperature_2m_mean", []),
        "temperature_2m_max":   daily.get("temperature_2m_max", []),
        "temperature_2m_min":   daily.get("temperature_2m_min", []),
        "precipitation_sum":    daily.get("precipitation_sum", []),
        "weathercode":          daily.get("weathercode", [])
    })
    df["weather"] = df["weathercode"].apply(lambda c: categorize_weather(int(c)))

    return func.HttpResponse(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False),
        status_code=200,
        mimetype="application/json"
    )
