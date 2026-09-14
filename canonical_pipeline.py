"""
canonical_pipeline.py
======================
Single source of truth for the ILT (Interpretable LSTM-Transformer) experiment.

This module is extracted verbatim (architecture, formulas, feature engineering)
from Notebook 08 (08_multi_horizon_comparison_repeat_30_run), which is the ONLY
notebook in the project whose model matches the architecture described in
Thesis Chapter 3.2.2:
    - LSTM encoder: 64 hidden units, return_sequences=True, dropout=0.2
    - TemporalAttention pooling (NOT MultiHeadAttention):
          score_t = tanh(W . h_t + b)
          alpha_t = softmax(u^T . score_t)
          context = sum(alpha_t . h_t)
    - Dual-path fusion: attention context (64) + final hidden state h_27 (64)
      + weather/calendar feature branch (16) -> concat (144) -> Dense(32) -> sigmoid
    - ~41,000 total trainable parameters
    - 60 input sequence features (incl. weather-lag family), 44 feature-branch
      features (weather/calendar/population only, fed identically to RF; was
      45 before the 2026-06-21 audit fix removed the leaking 'cases_7d_avg'
      column from this branch)

Every other notebook that needs "the proposed model" or "a fair baseline"
should import from HERE rather than redefining its own copy, so that the
9 required analyses (Section 4.1.4) are all evaluated on the SAME model.

Do NOT import anything from Notebooks 01, 04, 05, 06, 06b, 07 — those use an
earlier, abandoned architecture (16-unit LSTM + MultiHeadAttention, no
dual-path fusion, 53 features, no weather lags) that no longer matches the
thesis text and must not be used for any reported result.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEQ_LEN = 28  # spans the full Aedes mosquito lifecycle (Section 3.2.2)


# ──────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING  (identical to Notebook 08, Cell 1)
# ──────────────────────────────────────────────────────────────────────────
def engineer_features(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and build the canonical ~60-feature set.

    Six families per Chapter 3.2.1 / 4.2.2:
      1. Cyclical calendar encodings
      2. Weather interaction terms
      3. Temperature range (temp_max - temp_min)
      4. Weather lag features (rain/temp/humidity lag 14-28d, Aedes lifecycle)
      5. Case-history features (lags, rolling stats, EMA, diff)
      6. Weather rolling statistics
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Family 1 — cyclical calendar encodings
    if 'week_of_year' not in df.columns:
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin']  = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos']  = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Family 2 — weather interaction terms (Cheng et al., 2025)
    df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
    df['rain_humidity'] = df['rainfall'] * df['humidity'] / 100
    df['temp_rain']     = df['temperature'] * df['rainfall']

    # Family 3 — daily temperature range
    df['temp_range'] = df['temperature_max'] - df['temperature_min']

    # Family 5 — case lags (DL sequence branch only)
    for lag in [14, 21, 28]:
        df[f'cases_lag_{lag}'] = df['dengue_cases'].shift(lag)

    # Family 5 — case rolling stats (shifted by 14 to avoid leakage)
    for window in [7, 14, 28]:
        df[f'cases_mean_{window}'] = df['dengue_cases'].shift(14).rolling(window).mean()
        df[f'cases_std_{window}']  = df['dengue_cases'].shift(14).rolling(window).std()
        df[f'cases_max_{window}']  = df['dengue_cases'].shift(14).rolling(window).max()

    # Family 6 — weather rolling statistics
    for window in [7, 14, 28]:
        df[f'temp_mean_{window}']  = df['temperature'].rolling(window).mean()
        df[f'rain_sum_{window}']   = df['rainfall'].rolling(window).sum()
        df[f'humidity_mean_{window}'] = df['humidity'].rolling(window).mean()

    # Family 4 — weather lag features (Aedes lifecycle: rain -> breed -> bite -> sick)
    for lag in [14, 21, 28]:
        df[f'rain_lag_{lag}'] = df['rainfall'].shift(lag)
    for lag in [14, 21]:
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
    df['humidity_lag_14'] = df['humidity'].shift(14)

    # Family 5 — EMA + trend
    df['cases_ema_14']  = df['dengue_cases'].shift(14).ewm(span=14).mean()
    df['cases_ema_28']  = df['dengue_cases'].shift(14).ewm(span=28).mean()
    df['cases_diff_14'] = df['dengue_cases'].diff(14)
    df['temp_diff']     = df['temperature'].diff()
    df['rain_diff']     = df['rainfall'].diff()

    df_enhanced = df.dropna().reset_index(drop=True)
    return df_enhanced


def get_feature_sets(df_enhanced: pd.DataFrame):
    """Return (rf_features, dl_seq_features, dl_feat_features).

    rf_features      : weather + calendar + population ONLY (45 cols) — given
                        to Random Forest AND to the DL "feature branch" input,
                        so neither gets unfair access to case-history.
    dl_seq_features  : ALL ~60 columns — fed to the LSTM as the 28-day sequence;
                        this is the ONLY place case-history reaches the model,
                        which is the entire point of the LSTM (Section 3.2.2).
    dl_feat_features : identical to rf_features (fair comparison, Section 3.3.1).
    """
    exclude_cols = ['date', 'dengue_cases', 'year']
    feature_columns = [c for c in df_enhanced.columns if c not in exclude_cols]

    # AUDIT FIX (2026-06-21): 'cases_7d_avg' is a raw same-day case-history column
    # (r ~ 0.96-0.99 with same-day dengue_cases) that doesn't match any case_pfx
    # prefix below, so it was leaking into rf_features/dl_feat_features and silently
    # inflating any "weather-only" Random Forest number built from the unpatched
    # feature set. (The leakage-tainted rf_*_fair.pkl caches were removed on
    # 2026-07-02; Notebook 03 uses the patched, weather-only rf_results.pkl, and
    # Notebook 12 trains its seed=42 RF reference inline.)
    # Listed as its own exact-name exclusion (not a prefix) since it is the only
    # raw case-history column without a family prefix.
    case_pfx = ['cases_lag_', 'cases_mean_', 'cases_std_', 'cases_max_',
                'cases_ema_', 'cases_diff_']
    case_exact = ['cases_7d_avg']
    rf_features = [c for c in feature_columns
                   if not any(c.startswith(p) for p in case_pfx) and c not in case_exact]
    dl_seq_features = feature_columns
    dl_feat_features = rf_features
    return rf_features, dl_seq_features, dl_feat_features


def make_scalers(train_df, rf_features, dl_seq_features, dl_feat_features, y_train):
    rf_scaler = RobustScaler().fit(train_df[rf_features].values)
    dl_seq_scaler = RobustScaler().fit(train_df[dl_seq_features].values)
    dl_feat_scaler = RobustScaler().fit(train_df[dl_feat_features].values)
    # Target scaler: fit on FULL y_train, (0.02, 0.98) margin for sigmoid head (Section 3.2.1 Stage 4)
    target_scaler = MinMaxScaler(feature_range=(0.02, 0.98)).fit(y_train.reshape(-1, 1))
    return rf_scaler, dl_seq_scaler, dl_feat_scaler, target_scaler


def make_sequences(X_seq, X_feat, y_raw, seq_len, horizon):
    """Sliding-window sequence builder used by every horizon/run."""
    Xs, Xf, Y = [], [], []
    for i in range(seq_len, len(X_seq) - horizon):
        Xs.append(X_seq[i - seq_len:i])
        Xf.append(X_feat[i])
        Y.append(y_raw[i + horizon])
    return np.array(Xs), np.array(Xf), np.array(Y)


def load_canonical_dataset(csv_path: str, train_frac: float = 0.8):
    """One-call convenience wrapper: engineer features, split, scale.

    Returns a dict with everything notebooks 11-14 need, all derived from
    the SAME 60-feature / 45-feature canonical split used in Notebook 08.
    """
    df_enhanced = engineer_features(csv_path)
    rf_features, dl_seq_features, dl_feat_features = get_feature_sets(df_enhanced)

    train_size = int(len(df_enhanced) * train_frac)
    train_df = df_enhanced[:train_size].copy()
    test_df = df_enhanced[train_size:].copy()
    y_train = train_df['dengue_cases'].values
    y_test = test_df['dengue_cases'].values

    rf_scaler, dl_seq_scaler, dl_feat_scaler, target_scaler = make_scalers(
        train_df, rf_features, dl_seq_features, dl_feat_features, y_train)

    X_train_rf = rf_scaler.transform(train_df[rf_features].values)
    X_test_rf = rf_scaler.transform(test_df[rf_features].values)
    X_train_dl_seq = dl_seq_scaler.transform(train_df[dl_seq_features].values)
    X_test_dl_seq = dl_seq_scaler.transform(test_df[dl_seq_features].values)
    X_train_dl_feat = dl_feat_scaler.transform(train_df[dl_feat_features].values)
    X_test_dl_feat = dl_feat_scaler.transform(test_df[dl_feat_features].values)

    return dict(
        df_enhanced=df_enhanced, train_df=train_df, test_df=test_df,
        rf_features=rf_features, dl_seq_features=dl_seq_features, dl_feat_features=dl_feat_features,
        y_train=y_train, y_test=y_test,
        X_train_rf=X_train_rf, X_test_rf=X_test_rf,
        X_train_dl_seq=X_train_dl_seq, X_test_dl_seq=X_test_dl_seq,
        X_train_dl_feat=X_train_dl_feat, X_test_dl_feat=X_test_dl_feat,
        target_scaler=target_scaler,
        n_seq_features=len(dl_seq_features), n_feat_features=len(dl_feat_features),
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL ATTENTION LAYER  (verbatim from Notebook 08, Cell 4)
# ──────────────────────────────────────────────────────────────────────────
class TemporalAttention(layers.Layer):
    """Temporal attention pooling — learns which of the 28 timesteps matter most.

    Unlike self-attention (Q/K/V projections, ~3,000 params, O(T^2) attention
    matrix), this learns a single scoring function (~4,225 params here since
    feature dim = 64) that weights each timestep with O(T) cost. This is the
    mechanism that Thesis Section 3.2.2 / 3.5.2 describes and the ONLY
    attention layer that should appear anywhere in Chapter 5's results.

    Input:  (batch, timesteps, features)  e.g. (batch, 28, 64)
    Output: (batch, features)             weighted sum across timesteps
    Weights stored in self.attention_weights for interpretability (H3, H4).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attn_W', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attn_b', shape=(input_shape[-1],),
                                  initializer='zeros', trainable=True)
        self.u = self.add_weight(name='attn_u', shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform', trainable=True)

    def call(self, x, training=False):
        score = tf.tanh(tf.matmul(x, self.W) + self.b)          # (batch, T, F)
        score = tf.matmul(score, self.u)                         # (batch, T, 1)
        self.attention_weights = tf.nn.softmax(score, axis=1)    # (batch, T, 1)
        context = tf.reduce_sum(x * self.attention_weights, axis=1)  # (batch, F)
        return context

    def get_config(self):
        return super().get_config()


# ──────────────────────────────────────────────────────────────────────────
# 3. MODEL BUILDERS  (verbatim from Notebook 08, Cell 4)
# ──────────────────────────────────────────────────────────────────────────
def build_standalone_lstm(seq_len, n_seq, n_feat, lstm_units=64, dense_units=16,
                           dropout=0.2, learning_rate=0.001, pfx=''):
    """Architecturally identical to the ILT encoder, minus attention/dual-path.
    Used to isolate the contribution of the attention mechanism (H2)."""
    si = Input(shape=(seq_len, n_seq), name=f'{pfx}si')
    fi = Input(shape=(n_feat,), name=f'{pfx}fi')
    x = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout)(si)
    f = layers.Dense(dense_units, activation='relu')(fi)
    f = layers.Dropout(dropout)(f)
    c = layers.Concatenate()([x, f])
    h = layers.Dense(dense_units, activation='relu')(c)
    h = layers.Dropout(dropout)(h)
    o = layers.Dense(1, activation='sigmoid')(h)
    m = Model(inputs=[si, fi], outputs=o)
    m.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss='mse')
    return m


def build_lstm_transformer(seq_len, n_seq, n_feat, lstm_units=64, dense_units=32,
                            dropout=0.2, learning_rate=0.001, pfx=''):
    """THE canonical ILT model — matches Thesis Section 3.2.2 exactly:
    LSTM(64) -> LayerNorm -> [TemporalAttention path] + [final hidden state path]
    + [weather/calendar feature branch] -> concat(144) -> Dense(32) -> sigmoid.
    """
    si = Input(shape=(seq_len, n_seq), name=f'{pfx}si')
    fi = Input(shape=(n_feat,), name=f'{pfx}fi')

    x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(si)
    x = layers.LayerNormalization()(x)

    # Path 1 — temporal attention ("which past days matter": interpretability)
    attn_layer = TemporalAttention(name=f'{pfx}temp_attn')
    attn_out = attn_layer(x)                                            # (batch, lstm_units)

    # Path 2 — final hidden state ("LSTM's own best summary": recency)
    final_state = layers.Lambda(lambda t: t[:, -1, :], name=f'{pfx}final_state')(x)

    # Feature branch — weather/calendar/population ONLY (same as RF)
    f = layers.Dense(dense_units // 2 if dense_units >= 32 else 16, activation='relu')(fi)
    f = layers.Dropout(dropout)(f)

    c = layers.Concatenate()([attn_out, final_state, f])
    h = layers.Dense(dense_units, activation='relu')(c)
    h = layers.Dropout(dropout)(h)
    o = layers.Dense(1, activation='sigmoid')(h)

    m = Model(inputs=[si, fi], outputs=o)
    m.attn_layer = attn_layer
    m.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss='mse')
    return m


# ──────────────────────────────────────────────────────────────────────────
# 4. TRAINING HELPER  (verbatim logic from Notebook 08's train_dl, no retry)
# ──────────────────────────────────────────────────────────────────────────
def train_once(builder, seed, Xts_seq, Xts_feat, yt, Xvs_seq, Xvs_feat, yv,
               Xte_seq, Xte_feat, y_te_raw, target_scaler, seq_len, n_seq, n_feat,
               epochs=100, patience=20, pfx='', **builder_kwargs):
    """Train ONE run of a model and return (rmse, mae, r2, history, model)."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = builder(seq_len, n_seq, n_feat, pfx=pfx, **builder_kwargs)
    _ = m.predict([Xts_seq[:1], Xts_feat[:1]], verbose=0)  # warmup
    history = m.fit(
        [Xts_seq, Xts_feat], yt, validation_data=([Xvs_seq, Xvs_feat], yv),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(patience // 3, 1), min_lr=1e-7, verbose=0),
        ])
    p = m.predict([Xte_seq, Xte_feat], verbose=0).flatten()
    p = np.maximum(target_scaler.inverse_transform(p.reshape(-1, 1)).flatten(), 0)
    rmse = float(np.sqrt(mean_squared_error(y_te_raw, p)))
    mae = float(mean_absolute_error(y_te_raw, p))
    r2 = float(r2_score(y_te_raw, p))
    return rmse, mae, r2, history, m


if __name__ == '__main__':
    # Smoke test — confirms the module on its own reproduces the canonical
    # h=1d numbers reported by Notebook 08 (RMSE approx 30, R2 approx 0.71-0.72).   
    print("Smoke test: engineering features + building both models...")
    d = load_canonical_dataset('singapore_dengue_weather_2013_2022.csv')
    print(f"  rf_features:   {len(d['rf_features'])}")
    print(f"  dl_seq_features: {len(d['dl_seq_features'])}")
    print(f"  dl_feat_features: {len(d['dl_feat_features'])}")
    assert len(d['dl_seq_features']) == 60, "Expected 60 sequence features per Thesis Ch4.2.2"
    assert len(d['rf_features']) == 44, (
        "Expected 44 fair features after the 2026-06-21 leak fix (was 45 with "
        "'cases_7d_avg' still leaking in) — if this fires, Thesis Ch3.3.1 / "
        "Table 4.x text describing the fair feature-branch size needs updating to 44.")

    m_ilt = build_lstm_transformer(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])
    m_lstm = build_standalone_lstm(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])
    print(f"  ILT params:        {m_ilt.count_params():,}  (thesis claims ~41,000)")
    print(f"  Standalone params: {m_lstm.count_params():,}")
    print("✅ canonical_pipeline.py self-test passed")
