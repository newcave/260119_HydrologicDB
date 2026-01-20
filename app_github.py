import datetime as dt
from typing import Iterable

import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

BASE_URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw"
STATION_LIST_URL = f"{BASE_URL}/wl_dubwlobs"
HOURLY_WATER_LEVEL_URL = f"{BASE_URL}/wl_hrdata"
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)


def _find_first_list(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, list):
                        return nested
    raise ValueError("Unexpected response shape; cannot find list payload.")


def _safe_to_datetime(value: str) -> dt.datetime | None:
    for fmt in ("%Y%m%d%H", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


@st.cache_data(show_spinner=False)
def fetch_station_list() -> pd.DataFrame:
    response = requests.get(STATION_LIST_URL, params={"output": "json"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return pd.json_normalize(_find_first_list(payload))


@st.cache_data(show_spinner=False)
def fetch_hourly_water_level(obscd: str, start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(
        HOURLY_WATER_LEVEL_URL,
        params={
            "obscd": obscd,
            "startdt": start_date,
            "enddt": end_date,
            "output": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = pd.json_normalize(_find_first_list(payload))
    data["obscd"] = obscd
    return data


def normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [col for col in df.columns if "ymdh" in col.lower() or "ymd" in col.lower()]
    for col in candidate_cols:
        dt_values = df[col].astype(str).apply(_safe_to_datetime)
        if dt_values.notna().any():
            df = df.assign(parsed_datetime=dt_values)
            break
    return df


def build_lagged_dataset(
    df: pd.DataFrame, target_col: str, lag_hours: int
) -> tuple[pd.DataFrame, pd.Series]:
    if "parsed_datetime" in df.columns:
        df = df.sort_values("parsed_datetime")
    target = pd.to_numeric(df[target_col], errors="coerce")
    lagged = pd.DataFrame({f"{target_col}_lag_{i}h": target.shift(i) for i in range(1, lag_hours + 1)})
    dataset = lagged.join(target.rename(target_col)).dropna()
    return dataset.drop(columns=[target_col]), dataset[target_col]


def get_numeric_candidates(df: pd.DataFrame) -> list[str]:
    numeric_df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    candidates = [
        col
        for col in numeric_df.columns
        if numeric_df[col].notna().any()
    ]
    return candidates


def summarize_metrics(y_true: pd.Series, y_pred: Iterable[float]) -> dict:
    mse_value = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": mse_value**0.5,
    }


st.set_page_config(page_title="Uiryeong WAMIS Water Level", layout="wide")

st.title("의령 WAMIS 수위 데이터 조회 및 GBM/RF 해석")

st.markdown(
    """
이 앱은 WAMIS OpenAPI에서 의령 지역 수위 데이터를 조회하고,
8:2 학습/검증 분할로 Gradient Boosting과 Random Forest 모델을 비교합니다.
"""
)

st.markdown(
    """
#### 코드 변경 소개
- 문자열로 들어오는 수치 데이터를 자동으로 숫자형으로 변환합니다.
- 숫자형 후보를 찾지 못해 모델링이 중단되는 문제를 개선했습니다.
"""
)

with st.sidebar:
    st.header("조회 설정")
    default_start = dt.date.today() - dt.timedelta(days=7)
    start_date = st.date_input("시작일", value=default_start)
    end_date = st.date_input("종료일", value=dt.date.today())
    station_filter = st.text_input("관측소 필터", value="의령")
    fetch_button = st.button("관측소 목록 불러오기")

    st.header("모델 설정")
    test_size = st.slider("검증 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)

if fetch_button:
    stations = fetch_station_list()
    st.subheader("관측소 목록")
    if station_filter:
        mask = stations.apply(lambda row: station_filter in str(row.values), axis=1)
        stations = stations[mask]
    st.dataframe(stations, use_container_width=True)

    if stations.empty:
        st.warning("관측소가 없습니다. 필터를 변경해주세요.")
    else:
        st.info("관측소 코드(obscd)를 선택해서 수위 데이터를 조회하세요.")

st.subheader("수위 데이터 조회")
station_code = st.text_input("관측소 코드(obscd)", key="station_code")
load_data = st.button("수위 데이터 불러오기", key="load_data")

if "water_data" not in st.session_state:
    st.session_state.water_data = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if load_data and station_code:
    with st.spinner("데이터를 불러오는 중..."):
        data = fetch_hourly_water_level(
            station_code,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
        )
    data = normalize_datetime_columns(data)
    st.session_state.water_data = data

if st.session_state.water_data is not None:
    data = st.session_state.water_data
    st.success(f"로드 완료: {len(data)} 행")
    st.dataframe(data, use_container_width=True)

    numeric_cols = get_numeric_candidates(data)
    if not numeric_cols:
        st.warning("수치형 컬럼이 없습니다. 모델링을 진행할 수 없습니다.")
    else:
        target_col = st.selectbox("예측 대상 컬럼", options=numeric_cols, key="target_col")
        lag_hours = st.number_input(
            "과거 시간(lag, hours)", min_value=1, max_value=72, value=72, key="lag_hours"
        )
        train_model = st.button("GBM/RF 모델 학습", key="train_model")
        if train_model:
            if "parsed_datetime" not in data.columns:
                st.warning("시간 컬럼(예: ymdh)을 찾지 못해 lag 특징을 만들 수 없습니다.")
            X, y = build_lagged_dataset(data, target_col, lag_hours)
            if X.empty:
                st.warning("모델 학습에 사용할 데이터가 부족합니다.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                )
                gbm = GradientBoostingRegressor(random_state=random_state)
                rf = RandomForestRegressor(random_state=random_state, n_estimators=300)

                gbm.fit(X_train, y_train)
                rf.fit(X_train, y_train)

                gbm_pred = gbm.predict(X_test)
                rf_pred = rf.predict(X_test)

                gbm_metrics = summarize_metrics(y_test, gbm_pred)
                rf_metrics = summarize_metrics(y_test, rf_pred)

                metrics_df = pd.DataFrame([gbm_metrics, rf_metrics], index=["GBM", "RF"])
                comparison = pd.DataFrame(
                    {
                        "실제값": y_test.values,
                        "GBM 예측": gbm_pred,
                        "RF 예측": rf_pred,
                    }
                )
                st.session_state.model_results = {
                    "metrics": metrics_df,
                    "comparison": comparison,
                }
                st.session_state.trained_models = {
                    "GBM": gbm,
                    "RF": rf,
                    "target_col": target_col,
                    "lag_hours": lag_hours,
                }

        if st.session_state.model_results:
            st.subheader("모델 성능")
            st.dataframe(st.session_state.model_results["metrics"], use_container_width=True)

            st.subheader("예측 비교")
            comparison = st.session_state.model_results["comparison"]
            st.dataframe(comparison.head(50), use_container_width=True)
            st.line_chart(comparison, use_container_width=True)

            st.subheader("모델 저장/불러오기")
            model_name = st.text_input("모델 이름", value="uiryeong_model", key="model_name")
            save_model = st.button("모델 저장", key="save_model")
            load_model = st.button("모델 불러오기", key="load_model")

            if save_model:
                if st.session_state.trained_models:
                    model_path = MODEL_DIR / f"{model_name}.pkl"
                    with model_path.open("wb") as f:
                        pickle.dump(st.session_state.trained_models, f)
                    st.success(f"저장 완료: {model_path}")
                else:
                    st.warning("저장할 학습 모델이 없습니다.")

            if load_model:
                model_path = MODEL_DIR / f"{model_name}.pkl"
                if model_path.exists():
                    with model_path.open("rb") as f:
                        st.session_state.trained_models = pickle.load(f)
                    st.success(f"불러오기 완료: {model_path}")
                else:
                    st.warning("해당 이름의 저장된 모델이 없습니다.")

            if st.session_state.trained_models:
                loaded_target = st.session_state.trained_models.get("target_col")
                loaded_lag = st.session_state.trained_models.get("lag_hours")
                st.caption(f"현재 로드된 모델: target={loaded_target}, lag={loaded_lag}h")
                apply_loaded = st.button("불러온 모델로 예측", key="apply_loaded")
                if apply_loaded:
                    if loaded_target not in data.columns:
                        st.warning("로드된 모델의 예측 대상 컬럼이 현재 데이터에 없습니다.")
                    else:
                        X_loaded, y_loaded = build_lagged_dataset(data, loaded_target, loaded_lag)
                        if X_loaded.empty:
                            st.warning("예측에 사용할 데이터가 부족합니다.")
                        else:
                            gbm_loaded = st.session_state.trained_models.get("GBM")
                            rf_loaded = st.session_state.trained_models.get("RF")
                            gbm_pred_loaded = gbm_loaded.predict(X_loaded)
                            rf_pred_loaded = rf_loaded.predict(X_loaded)
                            metrics_df = pd.DataFrame(
                                [
                                    summarize_metrics(y_loaded, gbm_pred_loaded),
                                    summarize_metrics(y_loaded, rf_pred_loaded),
                                ],
                                index=["GBM", "RF"],
                            )
                            comparison = pd.DataFrame(
                                {
                                    "실제값": y_loaded.values,
                                    "GBM 예측": gbm_pred_loaded,
                                    "RF 예측": rf_pred_loaded,
                                }
                            )
                            st.session_state.model_results = {
                                "metrics": metrics_df,
                                "comparison": comparison,
                            }
else:
    st.info("관측소 코드를 입력하고 수위 데이터를 불러오세요.")
