# function_app.py

import logging
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
from catboost import CatBoostRegressor

import azure.functions as func

# ---------------- 配置区 ----------------
LATITUDE = 35.6895
LONGITUDE = 139.6917
TIMEZONE = "Asia/Tokyo"
MODEL_DIR = "saved_models"

TARGET_COLUMNS = [
    'pale_ale_bottles', 'lager_bottles', 'ipa_bottles',
    'white_beer_bottles', 'dark_beer_bottles', 'fruit_beer_bottles'
]
FEATURE_COLUMNS = [
    'is_friday',
    'apparent_temperature_mean',
    'precipitation_sum',
    'shortwave_radiation_sum'
]

# 进程启动时加载一次模型
def load_models():
    models = {}
    for t in TARGET_COLUMNS:
        m = CatBoostRegressor()
        m.load_model(f"{MODEL_DIR}/{t}.cbm")
        models[t] = m
    return models

models = load_models()

# 创建 FunctionApp 实例（匿名访问）
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="GetForecast", methods=["GET", "POST"])
def get_forecast(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GetForecast triggered")

    # 1. 读取 date 参数（URL Query 或 JSON body）
    date = req.params.get("date")
    if not date:
        try:
            body = req.get_json()
            date = body.get("date")
        except:
            pass
    if not date:
        return func.HttpResponse(
            "请通过 ?date=YYYY-MM-DD 或 JSON {'date':'YYYY-MM-DD'} 提供起始日期",
            status_code=400
        )

    # 2. 拉天气
    try:
        start = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return func.HttpResponse("日期格式错误，应为 YYYY-MM-DD", status_code=400)

    end = start + timedelta(days=6)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily": ",".join([
            "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_sum", "shortwave_radiation_sum",
            "precipitation_probability_mean", "uv_index_max", "weathercode"
        ]),
        "timezone": TIMEZONE
    }
    r = requests.get(weather_url, params=params)
    r.raise_for_status()
    d = r.json()["daily"]

    df = pd.DataFrame({
        "date": d["time"],
        "apparent_temperature_mean": [
            (mx + mn)/2 for mx, mn in zip(
                d["apparent_temperature_max"],
                d["apparent_temperature_min"]
            )
        ],
        "precipitation_sum":             d["precipitation_sum"],
        "shortwave_radiation_sum":       d["shortwave_radiation_sum"],
        "precipitation_probability_mean":d["precipitation_probability_mean"],
        "uv_index_max":                  d["uv_index_max"],
        "weathercode":                   d["weathercode"],
    })
    df["is_friday"] = pd.to_datetime(df["date"]).dt.weekday.eq(4).astype(int)

    # 3. 预测
    X = df[FEATURE_COLUMNS]
    for t, m in models.items():
        df[t] = m.predict(X).tolist()

    # 4. 格式化输出
    out = []
    for _, row in df.iterrows():
        e = {
            "date": row["date"],
            "apparent_temperature_mean":      round(row["apparent_temperature_mean"], 1),
            "precipitation_probability_mean": round(row["precipitation_probability_mean"], 1),
            "uv_index_max":                   round(row["uv_index_max"], 1),
            "weather_code":                   int(row["weathercode"])
        }
        for t in TARGET_COLUMNS:
            e[t] = int(round(row[t]))
        out.append(e)

    return func.HttpResponse(
        json.dumps(out, ensure_ascii=False),
        status_code=200,
        mimetype="application/json"
    )
